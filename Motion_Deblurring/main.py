"""
main.py：运动去模糊任务（GoPro 数据集）训练/测试的程序入口。
整体逻辑与去雾任务的 main.py 一致，区别在默认参数：
  - num_epoch 默认 3000（去模糊需要跑大量 epoch）
  - save_freq / valid_freq 默认 100
  - 测试权重默认 irnext_gopro.pkl
  - 模型存档目录 results/IRNeXt/test，测试输出目录 results/<model_name>/GOPRO

运行方式（在 Motion_Deblurring 目录下）：
  python main.py --mode train --data_dir 你的数据集路径
  python main.py --mode test  --data_dir 你的数据集路径 --test_model irnext_gopro.pkl
"""
import os
import torch
import argparse
from torch.backends import cudnn
from models.IRNeXt import build_net
from train import _train
from eval import _eval

def main(args):
    cudnn.benchmark = True  # 开启 cuDNN 自动选优卷积算法

    # ---- 创建目录 ----
    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()   # 实例化 IRNeXt（去模糊版，默认 num_res=14）
    print(model)

    if torch.cuda.is_available():
        model.cuda()      # 搬上 GPU

    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='IRNeXt', type=str)
    parser.add_argument('--data_dir', type=str, default='')  # 数据集根目录

    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)  # 运行模式

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3000)         # 训练轮数（很多）
    parser.add_argument('--print_freq', type=int, default=100)         # 每多少 iter 打印一次
    parser.add_argument('--num_worker', type=int, default=8)           # 数据加载进程数
    parser.add_argument('--save_freq', type=int, default=100)          # 每多少 epoch 存一次模型
    parser.add_argument('--valid_freq', type=int, default=100)         # 每多少 epoch 在验证集测 PSNR
    parser.add_argument('--resume', type=str, default='')              # 续训起始模型

    # Test
    parser.add_argument('--test_model', type=str, default='irnext_gopro.pkl')  # 测试权重
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])  # 是否保存复原图

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', 'IRNeXt', 'test')    # 模型存档目录
    args.result_dir = os.path.join('results/', args.model_name, 'GOPRO') # 测试输出目录
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    # 把源码复制进存档目录做代码快照
    command = 'cp ' + 'models/layers.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'models/IRNeXt.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'train.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'main.py ' + args.model_save_dir
    os.system(command)
    print(args)
    main(args)