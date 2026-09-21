"""
data_load.py：运动去模糊任务（GoPro 数据集）的数据加载器。
与去雾版的不同：
  - 目录为 blur/（模糊图）与 sharp/（清晰图），训练用 PairCompose(裁剪256+翻转+转张量)。
  - 测试与验证用的是 valid/ 子集（而非 test/）。
  - 真值文件名由 haz y 图名把 'blur' 替换成 'gt' 得到（GoPro 的命名约定）。
  - __getitem__ 用 'sharp' 目录与 .replace('blur','gt') 配对。
目录结构：train/{blur,sharp}、valid/{blur,sharp}。
"""
import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        # 训练增强：随机裁剪256 + 随机水平翻转 + 转张量（成对变换）
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,           # 训练打乱
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'valid')   # 测试用 valid 子集
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'valid')),   # 验证也用 valid 子集
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    """去模糊数据集的 Dataset：给定 image_dir（含 blur/ 与 sharp/ 子目录），
    按索引返回一对 (模糊图, 清晰图) 张量。"""

    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))   # 列出所有模糊图
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))  # 模糊图
        # 清晰图：同目录 sharp/ 下，文件名把 'blur' 替换为 'gt'（GoPro 命名约定）
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx].replace('blur', 'gt')))

        if self.transform:   # 训练：成对增强
            image, label = self.transform(image, label)
        else:                # 测试/验证：仅转张量
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