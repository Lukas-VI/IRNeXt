
"""
IRNeXt 主网络定义（运动去模糊任务，用于 GoPro 数据集）。
整体是一个「三尺度编解码 U 形卷积网络」，用于图像复原（去模糊）。
它与普通 U-Net 的不同点在于:
  1. 编码器/解码器的每一级由多个连续的 ResBlock 堆叠而成（EBlock / DBlock），
     EBlock 负责逐级下采样提取深层特征，DBlock 负责逐级上采样重建清晰图。
  2. 引入 SCM（空间补偿模块）和 FAM（特征融合模块），
     用于把“下采样后的原图”的多尺度细节补偿给主干网络，增强高频信息的保留。
  3. 网络输出是一个列表 outputs，包含三个尺度的复原预测图，
     训练时让每个尺度分别与对应下采样比例的真值(gt)做损失约束（深层监督 multi-scale supervision），
     推理/验证时只取 outputs[2]（原始分辨率那一张）。

该文件只负责“搭网络”，网络的基础算子（BasicConv / ResBlock / DeepPoolLayer /
dynamic_filter 等）都定义在同目录下的 layers.py 中，通过 `from .layers import *` 导入。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class EBlock(nn.Module):
    """编码器块（Encoder Block）：堆叠多个 ResBlock，作为编码器某一尺度上的特征提取主干。
    当之后接 stride=2 的下采样时，该块相当于在该分辨率上把特征“做深”。"""

    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        # 前 num_res-1 个是普通 ResBlock（不加深池化滤波）
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        # 最后一个 ResBlock 打开 filter=True，即内部会嵌入 DeepPoolLayer 多尺度动态滤波
        layers.append(ResBlock(out_channel, out_channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # 输入输出通道相同，直接顺序流过这一串 ResBlock
        return self.layers(x)


class DBlock(nn.Module):
    """解码器块（Decoder Block）：结构与 EBlock 完全相同。
    之所以分开命名，是为了在代码/论文里区分它服务于解码路径（重建），
    输出同样与原输入通道一致，可与编码器特征做加法/拼接。"""

    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        # 同样：num_res-1 个普通 ResBlock + 1 个带 DeepPoolLayer 的 ResBlock
        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    """SCM = Spatial Compensation Module（空间/细节补偿模块）。
    作用：从“下采样后的输入原图”里额外提取一组特征，供主干网络用来补偿
    在降采样过程中丢失的细节（主要是高频信息）。
    注意它吃的不是主干中间特征，而是直接吃下采样的 RGB 原图（3 通道），
    因此第一层是 BasicConv(3, ...)，把原图像素信息转换到特征空间。"""

    def __init__(self, out_plane):
        super(SCM, self).__init__()

        # 通道先降(make 1/4 → 1/2)再升(→ out_plane)，用 3x3 与 1x1 卷积交替，
        # 是一种轻量特征提取。最后 InstanceNorm 做归一化（去雾常见，避免对小 batch 敏感）。
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    """FAM = Feature Aggregation/Attention Module（这里实现为特征融合模块）。
    输入 x1、x2 两个同尺寸但不同来源的特征图，先把它们在通道维拼接，
    再用一个 3x3 卷积压缩回原通道数，得到融合后的特征。"""

    def __init__(self, channel):
        super(FAM, self).__init__()

        # channel*2 → channel：拼接后的通道数是原来的 2 倍，这里压缩回来
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        # dim=1 是通道维(N,C,H,W)，把两份特征沿通道拼接
        return self.merge(torch.cat([x1, x2], dim=1))

class IRNeXt(nn.Module):
    """IRNeXt 主干网络。
    输入 3 通道 RGB（直觉上 0~1），输出 3 个不同尺度的复原图组成的列表。
    三个尺度：原始尺寸(×1)、1/2 尺寸、1/4 尺寸。

    整体数据流（以输入 256x256 为例，详见 forward）：
      256 →(下采样) 128 →(下采样) 64 每层过 Encoder 编码，
      然后又 64 → 128 → 256 每层过 Decoder 反卷积重建，
      在每一级解码恢复分辨率后，都“长”出一张该分辨率的复原图（ConvsOut），
      并把原图在下采样后的拷贝(x_2 / x_4)直接加入对应预测（残差式输出）。"""

    def __init__(self, num_res=14):
        # 去模糊任务这里默认 num_res=14（去雾 ITS/OTS 为 4），
        # 即每个 EBlock/DBlock 内堆叠更多的 ResBlock，使网络更深、容量更大。
        super(IRNeXt, self).__init__()

        # 最底层特征通道数；之后每级缩放 2 倍(32→64→128)
        base_channel = 32
        # 编码器 3 级：通道分别 32 / 64 / 128，对应 3 个空间尺度
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        # feat_extract 完成尺度变换：
        #   [0][1][2]：stride=2 卷积 → 下采样（分辨率减半，通道加倍）
        #   [3][4]：stride=2 转置卷积 → 上采样（分辨率加倍，通道减半）
        #   [5]：最后把特征映射回 3 通道图
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        # 解码器 3 级，通道顺序与编码器相反（128→64→32）
        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        # 上采样重建时，先把“解码器输出”与“编码器同尺度特征(res1/res2)”按通道拼接，
        # 再用 1x1 卷积把拼接后的双倍通道压回单倍，等价于跳跃连接融合。
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        # ConvsOut 在解码中间层把特征图投影成三通道的复原图（输出候选）
        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        # FAM1/FAM2 融合“分支特征”与“SCM 补偿特征”；SCM1/SCM2 从下采样原图提取补偿
        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        # ---- 先对输入原图做 1/2 与 1/4 下采样，并提取 SCM 补偿特征 ----
        x_2 = F.interpolate(x, scale_factor=0.5)   # 1/2 尺寸原图
        x_4 = F.interpolate(x_2, scale_factor=0.5) # 1/4 尺寸原图
        z2 = self.SCM2(x_2)  # 从 1/2 图提取的补偿特征（128 尺度）
        z4 = self.SCM1(x_4)  # 从 1/4 图提取的补偿特征（64 尺度）

        outputs = list()
        # 256：第一阶段（原始分辨率）
        x_ = self.feat_extract[0](x)               # 3→32 通道，尺寸不变
        res1 = self.Encoder[0](x_)                  # 编码器第 0 级，输出 256 尺度特征
        # 128：下采样
        z = self.feat_extract[1](res1)              # stride=2，降到 128,通道 64
        z = self.FAM2(z, z2)                        # 与 SCM 补偿特征在 128 尺度融合
        res2 = self.Encoder[1](z)                   # 编码器第 1 级
        # 64：再下采样
        z = self.feat_extract[2](res2)              # stride=2，降到 64,通道 128
        z = self.FAM1(z, z4)                        # 与 1/4 图的补偿融合
        z = self.Encoder[2](z)                      # 编码器第 2 级（最深层）

        # ---- 解码路径，逐级上采样重建 ----
        z = self.Decoder[0](z)                      # 解码器第 0 级（64 尺度）
        z_ = self.ConvsOut[0](z)                    # 投影出 64 尺度复原图（对应 1/4 尺寸）
        # 128：上采样
        z = self.feat_extract[3](z)                 # 转置卷积上采样到 128
        outputs.append(z_+x_4)                      # 第 0 个输出：64尺度预测 + 1/4 原图(残差)

        z = torch.cat([z, res2], dim=1)             # 沿通道拼解码器输出与编码器同尺度特征(跳跃连接)
        z = self.Convs[0](z)                        # 1x1 卷积压缩通道 128->64
        z = self.Decoder[1](z)                      # 解码器第 1 级
        z_ = self.ConvsOut[1](z)                    # 128 尺度复原图（对应 1/2 尺寸）
        # 256：
        z = self.feat_extract[4](z)                 # 上采样回到 256
        outputs.append(z_+x_2)                      # 第 1 个输出：128尺度预测 + 1/2 原图(残差)

        z = torch.cat([z, res1], dim=1)             # 拼接第一级编码器特征(跳跃连接)
        z = self.Convs[1](z)                        # 1x1 卷积压通道 64->32
        z = self.Decoder[2](z)                      # 解码器第 2 级
        z = self.feat_extract[5](z)                 # 特征投影回 3 通道，得到原始分辨率复原图
        outputs.append(z+x)                         # 第 2 个输出：原始尺寸预测 + 原图(残差)

        return outputs


def build_net():
    """构建网络的工厂函数，供 main / train 等直接调用生成模型实例。"""
    return IRNeXt()
