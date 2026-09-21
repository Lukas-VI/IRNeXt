"""
IRNeXt 的基础网络算子库（运动去模糊版）。
与去雾版不同的地方主要是：
  - 新增了一个自定义 AvgPool2d 模块，用于做一种“可缩放的、基于积分图(累计和)的
    高效平均池化”。在 dynamic_filter 中被用作 gap（整图均值 / 低频近似），
    base_size=120 表示该池化的“基准感受野”为 120x120（随输入分辨率按比例缩放）。
  - dynamic_filter 内部使用上面的 AvgPool2d 而非 torch 的 AdaptiveAvgPool2d。
其余模块（BasicConv / ResBlock / DeepPoolLayer / dynamic_filter）的作用与去雾版一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



# Borrowed from ''Improving image restoration by revisiting global information aggregation''
# --------------------------------------------------------------------------------
# （保留原作者注释）以下 AvgPool2d 模块借自论文《Improving image restoration by
# revisiting global information aggregation》，是一种更精细的自适应全局/区域池化。
# 训练时的参考输入尺寸，用于按比例把 base_size 从训练分辨率换算到当前分辨率。
train_size = (1,3,256,256)

class AvgPool2d(nn.Module):
    """可缩放平均池化：无需指定固定 kernel，而是依据输入分辨率与 base_size
    自动计算一个与输入尺寸成正比的池化窗口，并用积分图(前缀和)快速求窗口内均值。
    fast_imp=True 时先用 stride 采样降低计算量再上采样回原尺寸（近似提速）；
    默认走积分图精确版本。auto_pad=True 时把输出 pad 回与输入一致的分辨率。"""

    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        self.fast_imp = fast_imp
        self.rs = [5,4,3,2,1]  # fast_imp 模式下可用的下采样比例
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )
           
    def forward(self, x):
        # 当没给定 kernel_size 而给了 base_size 时，把 base_size 按输入分辨率相对
        # 训练分辨率(256)的比例换算成实际的 kernel 大小（即池化窗口随输入一起缩放）。
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2]*self.base_size[0]//train_size[-2]
            self.kernel_size[1] = x.shape[3]*self.base_size[1]//train_size[-1]
            
            # fast_imp 模式下也按比例缩放允许的下采样比例上界
            self.max_r1 = max(1, self.rs[0]*x.shape[2]//train_size[-2])
            self.max_r2 = max(1, self.rs[0]*x.shape[3]//train_size[-1])

        if self.fast_imp:  # 快速近似分支：先按比例下采样，池化后再上采样回原尺寸
            h, w = x.shape[2:]
            if self.kernel_size[0]>=h and self.kernel_size[1]>=w:
                # 窗口不小于输入时直接全局池化到 1x1
                out = F.adaptive_avg_pool2d(x,1)
            else:
                r1 = [r for r in self.rs if h%r==0][0]  # 取能整除 h 的第一个采样率
                r2 = [r for r in self.rs if w%r==0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:,:,::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)  # 采样后用积分图
                n, c, h, w = s.shape
                # 用积分图的四角差求窗口内均值（前缀和技巧：区间和 = 两前缀之差）
                k1, k2 = min(h-1, self.kernel_size[0]//r1), min(w-1, self.kernel_size[1]//r2)
                out = (s[:,:,:-k1,:-k2]-s[:,:,:-k1,k2:]-s[:,:,k1:,:-k2]+s[:,:,k1:,k2:])/(k1*k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1,r2))  # 回采样原尺寸
        else:
            # 精确分支：直接在整个特征上做二维前缀和（积分图），
            # 再用四角差值计算每个 kernel 窗口内的均值，无需显式滑窗，计算最高效。
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)  # 先沿宽再沿高累计 = 二维前缀和
            s = torch.nn.functional.pad(s, (1,0,1,0))  # 前边各补一列/一行 0 以对齐
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])

            s1, s2, s3, s4 = s[:,:,:-k1,:-k2],s[:,:,:-k1,k2:], s[:,:,k1:,:-k2], s[:,:,k1:,k2:]
            out = s4+s1-s2-s3  # 四角组合 = 矩形内累加和
            out = out / (k1*k2)  # 除以窗口面积得到均值
    
        if self.auto_pad:
            # 把输出 padding 回与输入图相同的分辨率（偶数/奇数分别处理左右/上下）
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w)//2, (w - _w + 1)//2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
        
        return out

class BasicConv(nn.Module):
    """最基础的卷积块（同去雾版）：卷积(可选转置)/归一化/激活。"""

    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    """残差块（同去雾版），filter=True 时内部嵌入 DeepPoolLayer 多尺度动态滤波。"""

    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        # 残差短路连接
        return self.main(x) + x


class DeepPoolLayer(nn.Module):
    """DeepPoolLayer（同去雾版）：多尺度池化(8/4/2)特征聚合，捕捉多尺度上下文。"""

    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8,4,2]
        pools, convs, dynas = [],[],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            dynas.append(dynamic_filter(inchannels=k, kernel_size=3))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                # 逐级把上一尺度上采样 2 倍后累加到本尺度，再处理
                y = self.dynas[i](self.convs[i](self.pools[i](x)+y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl

class dynamic_filter(nn.Module):
    """dynamic_filter（动态滤波器，去模糊版）。
    与去雾版几乎一致，唯一的区别是这里的 gap 使用自定义 AvgPool2d(base_size=120)
    而不是 torch 的全局平均池化。也就是说低频近似项用“窗口随分辨率缩放的池化”得到，
    而非严格的整图均值，这样低频分量更贴合局部结构（对去模糊保留细节更关键）。
    其余低通+高通分支的机制完全相同。"""

    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Tanh()
    
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        # 与去雾版不同：这里用自定义的、带 120x120 基准窗口的 AvgPool2d 作为 gap
        self.gap = AvgPool2d(base_size=120)

        self.inside_all = nn.Parameter(torch.zeros(inchannels,1,1), requires_grad=True)
    def forward(self, x):
        
        identity_input = x
        low_filter = self.ap(x)          # 全局均值 → 预测低频滤波器系数
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        # unfold 展开成窗口并重组为分组形式，与滤波系数逐窗相乘做“分组动态卷积”
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
        low_filter = self.act(low_filter)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)  # 低频分量

        # 用可学习 inside_all 加权 low_part，并减去 gap 相关的整体均值项（低频归一化近似）
        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)

        out_low = out_low * self.lamb_l[None,:,None,None]   # 低通逐通道缩放
        out_high = (identity_input) * (self.lamb_h[None,:,None,None] + 1.)   # 高通(恒等)逐通道缩放

        return out_low + out_high

