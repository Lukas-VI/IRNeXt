"""
utils.py：整个项目共用的两个小工具类。
  - Adder：一个简单的「累加器」，用于累加若干次损失/指标并求平均值，
    便于在每个打印周期打印当前迭代的均值损失、以及每个 epoch 的均值损失。
  - Timer：计时器，记录两段代码之间的耗时；可按秒/分钟/小时三种单位返回。
  - check_lr：从 optimizer 的 param_groups 里取回当前学习率。
这两个类在 main / train / eval / valid 中都被 import 使用。
"""
import time
import numpy as np


class Adder(object):
    """累加器：把多次调用传入的数字累加起来，并提供平均值。"""

    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        # 清空累计，开始新一轮累加（如每打印一次后、每个 epoch 结束后）
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        # 每次“调用实例”都相当于往里加一个数值，如 adder(loss.item())
        self.count += 1
        self.num += num

    def average(self):
        # 返回目前累加的所有数值的平均值（count 为 0 时可能报 ZeroDivisionError）
        return self.num / self.count


class Timer(object):
    """计时器：tic() 记录开始时间，toc() 返回从 tic 到现在的耗时。"""

    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        # 根据 option 选择返回的计时单位（除数：秒 1 / 分钟 60 / 小时 3600）
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        # 返回从 tic 到现在的秒数，再除以单位换算得到所需单位下的数值
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    # 遍历 optimizer 的所有参数组（这里其实只需要取最后一个 / 唯一一个），返回当前学习率
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr
