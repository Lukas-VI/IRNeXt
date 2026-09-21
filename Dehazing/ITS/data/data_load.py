"""
data_load.py：去雾任务（ITS 数据集）的数据加载器。
负责从磁盘读取「有雾图/清晰真值图」的成对数据并封装成 DataLoader。
数据集目录结构约定（args.data_dir 下）：
  train/hazy/*.png   训练用有雾图
  train/gt/*.png     训练用清晰真值（文件名取 haz y 名字中 '_' 前部分）
  test/hazy/*.png    测试用有雾图（同样配对 gt/）
  test/gt/*.png
训练用 PairCompose 随机裁剪 256x256 + 随机水平翻转 + 转张量做增强；
测试/验证只读图并转张量（不打乱、不增强），测试模式额外返回文件名。
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
        # 训练时做随机裁剪(256) + 随机水平翻转 + 转张量
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
        shuffle=True,          # 训练打乱
        num_workers=num_workers,
        pin_memory=True        # 用固定内存加速 GPU 取数
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),   # is_test=True 会额外返回文件名
        batch_size=batch_size,                    # 测试逐张(batch=1)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test')),   # 验证集也直接用 test 子集
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    """去雾数据集的 Dataset：给定 image_dir（含 hazy/ 与 gt/ 子目录），
    按索引返回一对 (有雾图, 清晰图) 张量。"""

    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))  # 列出所有有雾图
        self._check_image(self.image_list)   # 校验扩展名合法
        self.image_list.sort()               # 排序保证顺序稳定、可复现
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx]))  # 有雾图
        # 真值文件名 = 文件名中去掉后缀中 '_数字'部分 + .png（约定配对命名）
        label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.png'))

        if self.transform:    # 训练：做成对增强
            image, label = self.transform(image, label)
        else:                 # 测试/验证：仅转张量
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]   # 测试还返回文件名，便于保存图片
            return image, label, name
        return image, label    # 训练/验证返回两元组

    @staticmethod
    def _check_image(lst):
        # 校验文件名后缀是否合法（png/jpg/jpeg）
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
