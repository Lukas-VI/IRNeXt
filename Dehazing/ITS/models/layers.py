"""
IRNeXt 的基础网络算子库（去雾版）。
这里定义了 IRNeXt 网络用到的所有“积木块”：
  - BasicConv：一个打包好的卷积模块（Conv2d/ConvTranspose2d + BN + 激活），
    是网络里最基本的单元。
  - ResBlock：带“短路连接”的残差块；可选地在内部镶嵌 DeepPoolLayer。
  - DeepPoolLayer：多尺度池化特征聚合模块（论文里的核心设计之一），
    把特征分别做 8/4/2 倍下采样池化，再逐级上采样累加回原分辨率，以捕捉多尺度上下文。
  - dynamic_filter：动态滤波器模块。用一个(几乎所有通道共享的)轻量预测器
    从全局池化特征生成空间变化的“低频滤波器”，并反作用于输入特征，
    再配合一个“高通分支”，实现低频/高频分离处理。

这些模块被同目录的 IRNeXt.py（主网络）所使用，`from .layers import *` 导入。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicConv(nn.Module):
    """最基础的卷积块 = 卷积(可选转置)/归一化/激活 打包在一起。
    带 transpose=True 时用转置卷积做上采样（特征图变大）；
    否则用普通卷积（stride=2 时即下采样）。bias 与 norm 同时打开时 bias 置 False
    （因为 BN 里已有平移项，再加 bias 冗余）。"""

    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        # 用“核大小//2”保证 padding，使卷积不改变分辨率（stride=1 时）
        padding = kernel_size // 2
        layers = list()
        if transpose:
            # 转置卷积要对应地把 padding 减 1，配合 stride 才能实现“2 倍上采样且有正确输出尺寸”
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            # 激活用 GELU 而非 ReLU（图像复原论文常用，曲线更平滑）
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    """残差块：main 分支为“Conv-GELU-(可选 DeepPool)-GELU-Conv”，
    再与输入 x 相加（短路连接）。filter=True 时在中间插入 DeepPoolLayer
    做多尺度动态滤波，这也是 IRNeXt 关键设计点；filter=False 时退化为普通残差块。"""

    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            # filter=True 时插入多尺度池化聚合模块，否则用恒等映射跳过
            DeepPoolLayer(in_channel, out_channel) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        # 残差连接：学习“残差”而不是整幅特征，利于训练稳定与梯度传播
        return self.main(x) + x


class DeepPoolLayer(nn.Module):
    """DeepPoolLayer：多尺度/多分辨率池化特征聚合。
    将输入特征分别用 AvgPool 下采样 8 倍、4 倍、2 倍，
    每条支路经过卷积 + 动态滤波后，再上采样回原尺寸并与输入逐像素相加；
    最后经过一层 gelu 与 1x1 卷积输出。
    目的：让网络在单个 ResBlock 内部就能看到多尺度上下文信息（金字塔式的感受野）。"""

    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8,4,2]   # 三条支路的下采样倍数
        pools, convs, dynas = [],[],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))     # 各自倍率的下采样
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))       # 下采样后做 3x3 卷积
            dynas.append(dynamic_filter(inchannels=k, kernel_size=3)) # 动态滤波
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False) # 聚合输出压缩通道

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                # 第一支：只在池化后特征上做卷积+动态滤波
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                # 后续支路：把上一支上采样 2 倍后的结果 y_up 加到本支池化结果上再处理，
                # 实现“逐级 2 倍上采样并在各尺度累加”，即把多尺度信息逐步拼回去
                y = self.dynas[i](self.convs[i](self.pools[i](x)+y_up))
            # 上采样回原始分辨率并累加到残差上（这也是“累加多尺度”的关键）
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                # 为下一支准备 2 倍放大版本的 y（注意这里 interpolate 用 scale_factor=2）
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl

class dynamic_filter(nn.Module):
    """dynamic_filter：动态滤波器（低频/高频分离处理）。
    核心思想：先从输入特征的全局平均池化结果生成一个「空间不变的低频滤波器系数」，
    作用到输入特征上得到低频分量 low_part；再让输入特征经过一个带学习参数
    (lamb_l / lamb_h / inside_all)的可学习加权，得到低通加权输出 out_low 与
    高通通路 out_high，最终合并成 out_low + out_high 返回。

    直观作用：让网络能自适应地为不同位置分配“更平滑”与“更锐利”的处理，
    从而更好地保留细节（高频）同时去除伪影（低频），这正是复原任务需要的。"""

    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        # 用 1x1 卷积从全局特征预测滤波器系数：输出 channel = group * k^2
        # 即把整幅图压缩成 group 组、每组分得 kernel 个数(k^2)的系数
        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Tanh()   # 系数压缩到 (-1,1) 附近，保证稳定的滤波系数

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # 可学习缩放参数：lamb_l 乘到低通输出上，lamb_h 乘到高通(恒等)分支上
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size//2)  # unfold 前做反射 padding 以对齐窗口

        self.ap = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，生成空间不变滤波系数
        self.gap = nn.AdaptiveAvgPool2d(1)      # 另一个“整图均值”(用于低频近似,见 forward)

        # 可学习常数（1,1,1），参与低频加权
        self.inside_all = nn.Parameter(torch.zeros(inchannels,1,1), requires_grad=True)

    def forward(self, x):
        identity_input = x   # 保留输入本身，作为高通/恒等通路
        # the Conv_{3x3} layer in eq.3 is included in DeepPoolLayer.convs
        # （保留原文注释）eq.3 中的 3x3 卷积由 DeepPoolLayer.convs 承担，这里只做系数预测
        low_filter = self.ap(x)        # 全局均值 → 1x1x1
        low_filter = self.conv(low_filter)  # 预测滤波器系数（通道数为 group*k^2）
        low_filter = self.bn(low_filter)     # 归一化

        n, c, h, w = x.shape
        # 用 unfold 把输入按 kernel_size 窗口展开，并重组为 [n, group, c//group, k^2, h*w]
        # 这是为下一步与滤波系数做逐窗口加权（相当于分组的动态卷积）
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        # 把系数 reshape 成 [n, c1/k^2, k^2, p*q] 再 unsqueeze 到第 2 维，用于和展开窗口相乘
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
        low_filter = self.act(low_filter)   # Tanh 压缩
        # 逐窗口相乘求和（dim=3 是 k^2 轴），再 reshape 回原尺寸得到“低频滤波结果”
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        # the variables here are slightly different from the paper: (code) --> (paper)
        # low_filter --> A (eq.3)
        # In Eq.7, X*A'= X*(A_{l} + WA_{h})
        #              = X*A_{l} + WX*(A - A_{l})
        #              = X*A_{l} + WX*A - WX*A_{l}
        #              = WX*A - X*A_{l}(W-1)
        # we substitute gap for A_{l} for simplicity, which is a coarser low-frequency filter
        #（保留原作者推导说明：用 X*A' 的展开式，以 gap 近似 A_{l}（更粗的低频滤波））
        # out_low 用可学习的 inside_all 对 low_part 做加权，并减去 gap(整图均值) 相关项，
        # 从而加强低频分量自身并抑制整体均值漂移
        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)

        out_low = out_low * self.lamb_l[None,:,None,None]  # 低通分支的逐通道缩放
        out_high = (identity_input) * (self.lamb_h[None,:,None,None] + 1.)  # 高通(恒等)分支缩放

        # 低通 + 高通合并输出
        return out_low + out_high

