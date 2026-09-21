"""
data_load.py：去雾任务（OTS 数据集）的数据加载器。
与 ITS 的 data_load.py 逻辑一致，但实现方式不同：
  - 不依赖 transforms 的 Pair* 类，而是直接在 Dataset.__getitem__ 里手动做
    随机裁剪(ps=256)与随机水平翻转。
  - OTS 训练真值图为 jpg 格式（ITS 为 png），测试/验证真值仍为 png。
  - 设置了 ImageFile.LOAD_TRUNCATED_IMAGES = True，允许加载“截断的/损坏的图片”，
    这是针对 OTS 大规模数据中存在个别坏图的处理。
目录结构：train/{hazy,gt}、test/{hazy,gt}。
"""
import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # 允许读入截断图，避免 OTS 中坏图导致崩溃

def train_dataloader(path, batch_size=64, num_workers=0):
    image_dir = os.path.join(path, 'train')

    dataloader = DataLoader(
        DeblurDataset(image_dir, ps=256),   # 训练时在 Dataset 内做 256 随机裁剪/翻转
        batch_size=batch_size,
        shuffle=True,                 # 训练打乱
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),   # is_test 返回文件名
        batch_size=batch_size,                    # 测试逐张
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test'), is_valid=True),   # 验证也用 test 子集
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader

import random
class DeblurDataset(Dataset):
    """去雾数据集的 Dataset（OTS 版）：手动实现随机裁剪/翻转增强。"""

    def __init__(self, image_dir, transform=None, is_test=False, is_valid=False, ps=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.is_valid = is_valid
        self.ps = ps   # 裁剪尺寸（ps=None 表示不裁剪）

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 统一转 RGB；有雾图
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx])).convert('RGB')
        # 真值：验证/测试用 png，训练用 jpg
        if self.is_valid or self.is_test:
            label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.png')).convert('RGB')
        else:
            label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.jpg')).convert('RGB')
        ps = self.ps

        if self.ps is not None:   # 训练：先转张量再随机裁剪 ps 大小
            image = F.to_tensor(image)
            label = F.to_tensor(label)

            hh, ww = label.shape[1], label.shape[2]   # 真值的高/宽

            # 随机取裁剪左上角 (rr, cc)，保证裁剪区域不越界
            rr = random.randint(0, hh-ps)
            cc = random.randint(0, ww-ps)

            # 对两张图按同一位置裁剪
            image = image[:, rr:rr+ps, cc:cc+ps]
            label = label[:, rr:rr+ps, cc:cc+ps]

            if random.random() < 0.5:   # 以 0.5 概率做水平翻转(通道维左右翻转 = 高维 flip，即第3维width)
                image = image.flip(2)
                label = label.flip(2)
        else:                         # 测试/验证：仅转张量（不裁剪）
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            name = self.image_list[idx]   # 测试返回文件名
            return image, label, name
        return image, label



    @staticmethod
    def _check_image(lst):
        # 校验扩展名合法性
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError