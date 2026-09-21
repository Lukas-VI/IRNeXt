"""
main.py：去雾任务 OTS 数据集训练/测试的程序入口。
整体逻辑与 ITS 的 main.py 完全一致，区别只在默认参数不同：
  - batch_size 默认 8（ITS 为 4）
  - num_epoch 默认 30（ITS 为 300，因为 ITS 数据量更大需要更多轮）
  - save_freq / valid_freq 默认 1（每个 epoch 都存档/验证）
  - 模型存档目录 results/IRNeXt/OTS/

运行方式（在 OTS 目录下）：
  python main.py --mode train --data_dir 你的数据集路径
  python main.py --mode test  --data_dir 你的数据集路径 --test_model 模型.pkl
"""
import os
import torch
import argparse
from torch.backends import cudnn
from models.IRNeXt import build_net
from train import _train
from eval import _eval


def main(args):
    # CUDNN
    cudnn.benchmark = True  # 开启 cuDNN 自动选择最优卷积算法，加速训练

    # ---- 创建目录 ----
    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()   # 实例化 IRNeXt 网络
    print(model)          # 打印网络结构

    if torch.cuda.is_available():
        model.cuda()      # 有 GPU 则搬上 GPU

    if args.mode == 'train':
        _train(model, args)   # 训练

    elif args.mode == 'test':
        _eval(model, args)    # 测试/评估


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='IRNeXt',type=str)
    parser.add_argument('--data_dir', type=str, default='')  # 数据集根目录
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)  # 运行模式

    # Train
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=30)          # 训练轮数
    parser.add_argument('--print_freq', type=int, default=100)        # 每多少 iter 打印一次
    parser.add_argument('--num_worker', type=int, default=8)          # 数据加载进程数
    parser.add_argument('--save_freq', type=int, default=1)           # 每多少 epoch 存一次模型
    parser.add_argument('--valid_freq', type=int, default=1)          # 每多少 epoch 在验证集测 PSNR
    parser.add_argument('--resume', type=str, default='')             # 续训起始模型文件


    # Test
    parser.add_argument('--test_model', type=str, default='')         # 测试用权重文件
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])  # 是否保存复原图

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', 'IRNeXt', 'OTS/')   # 模型存档目录
    args.result_dir = os.path.join('results/', args.model_name, 'test') # 测试输出目录
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    # 把源码文件复制进存档目录，作为代码快照便于复现
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