"""
data_augment.py：运动去模糊任务（GoPro）共用的「成对」数据增强变换类。
与去雾版的 data_augment.py 基本一致（所有变换同时对模糊图与清晰图施加同一操作），
仅额外把「成对垂直翻转」PairRandomVerticalFlip 注释掉了（作者未启用）。
"""
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):
    """成对随机裁剪：对 image 与 label 在同一位置裁剪同样大小的块。
    继承 torchvision 的 RandomCrop，在 __call__ 里结合两个输入。"""

    def __call__(self, image, label):

        # 若配置了 padding，先对两张图做相同的 padding
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        # 若要求需要时补齐且宽不足，则补宽
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        # 同理补高
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - label.size[1]), self.fill, self.padding_mode)

        # 由父类随机生成裁剪起始点与尺寸，对两张图用同一个参数裁剪
        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    """成对组合：把多个 Pair* 变换串联起来，同时作用在同一对 (image, label) 上。"""

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    """成对随机水平翻转：以概率 p 对两张图都做 hflip，否则都不做。"""

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


#class PairRandomVerticalFlip(transforms.RandomVerticalFlip):
#    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
#        if random.random() < self.p:
#            return F.vflip(img), F.vflip(label)
#        return img, label
# （上面一整段成对垂直翻转类被作者注释掉了，代码中并未启用，保留原注释状态）


class PairToTensor(transforms.ToTensor):
    """成对转张量：把两张 PIL 图同时转为 [0,1] 的 torch.Tensor。"""

    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)