"""
preprocessing.py：GoPro 数据集的预处理脚本。
GoPro 原始数据集按场景分子目录（train/xx/blur、train/xx/sharp...）。
本脚本把每个场景下的 blur/sharp 图片“展平”到一个扁平目录结构，
并用「场景名_」作前缀重命名，便于后续数据加载器直接按文件夹读取。
运行：python preprocessing.py --root_src 原始GoPro路径 --root_dst 输出路径
"""
import os
import argparse


def move(src, dst):
    """把 src 目录树下每个子场景的 blur/sharp 图移动到 dst/blur 与 dst/sharp。"""
    if not os.path.exists(dst):              # 建立目标根目录
        os.mkdir(dst)
    if not os.path.exists(os.path.join(dst, 'blur')):    # 建立 blur 子目录
        os.mkdir(os.path.join(dst, 'blur'))
    if not os.path.exists(os.path.join(dst, 'sharp')):   # 建立 sharp 子目录
        os.mkdir(os.path.join(dst, 'sharp'))

    folders = os.listdir(src)   # 一级子目录即各场景
    cnt = 0
    for f in folders:
        image_names = os.listdir(os.path.join(src, f, 'blur'))   # 该场景下所有模糊图

        for i in image_names:
            # 移动到扁平目录（blur 与 sharp 同名，不同目录），并加上场景名前缀避免重名
            os.rename(os.path.join(src, f, 'blur', i), os.path.join(dst, 'blur', f + '_' + i))
            os.rename(os.path.join(src, f, 'sharp', i), os.path.join(dst, 'sharp', f + '_' + i))
            cnt += 1
    print('%d images are moved' % cnt)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('--root_src', default='dataset/GOPRO_Large', type=str)  # 原始GoPro根目录
    parser.add_argument('--root_dst', default='dataset/GOPRO', type=str)        # 预处理输出目录

    args = parser.parse_args()

    if not os.path.exists(args.root_dst):
        os.mkdir(args.root_dst)

    # 分别处理 train 与 test 两个子集
    move(os.path.join(args.root_src, 'train'), os.path.join(args.root_dst, 'train'))
    move(os.path.join(args.root_src, 'test'), os.path.join(args.root_dst, 'test'))
