"""
main.py：整个去雾训练/测试流程的程序入口。
作用：
  1. 解析命令行参数（batch_size、learning_rate、epoch 数、数据类型等）；
  2. 构建 IRNeXt 网络模型（build_net）；
  3. 根据 `--mode` 决定走训练(_train)还是测试(_eval)；
  4. 在 `results/` 下准备模型存档目录，并把本项目的源码(模型/训练脚本)复制到
     存档目录，方便日后复盘“这个模型是用哪版代码训出来的”。
运行方式（在 ITS 目录下）：
  python main.py --mode train  --data_dir 你的数据集路径
  python main.py --mode test   --data_dir 你的数据集路径 --test_model 模型.pkl
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

    # ---- 创建各类目录（results 及模型存档目录/测试输出目录）----
    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()   # 实例化 IRNeXt 网络
    print(model)          # 打印网络结构，便于查看

    if torch.cuda.is_available():
        model.cuda()      # 有 GPU 就把模型搬到 GPU

    if args.mode == 'train':
        _train(model, args)   # 进入训练循环

    elif args.mode == 'test':
        _eval(model, args)    # 进入测试（评估/推理）流程


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='IRNeXt', type=str)

    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)  # 运行模式
    parser.add_argument('--data_dir', type=str, default='')  # 数据集根目录（含 train/test 子目录）

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)          # 训练轮数
    parser.add_argument('--print_freq', type=int, default=100)         # 每隔多少 iter 打印一次
    parser.add_argument('--num_worker', type=int, default=8)           # DataLoader 加载进程数
    parser.add_argument('--save_freq', type=int, default=10)           # 每隔多少 epoch 存一次模型
    parser.add_argument('--valid_freq', type=int, default=10)          # 每隔多少 epoch 在验证集上测 PSNR
    parser.add_argument('--resume', type=str, default='')              # 若要接着训练，给定模型文件路径


    # Test
    parser.add_argument('--test_model', type=str, default='')          # 测试用的模型权重文件
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])  # 是否保存复原图

    args = parser.parse_args()
    # 模型存档位置：results/IRNeXt/ITS/
    args.model_save_dir = os.path.join('results/', 'IRNeXt', 'ITS/')
    # 测试输出图位置：results/<model_name>/test
    args.result_dir = os.path.join('results/', args.model_name, 'test')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    # 把用到的源码文件复制进存档目录，留作训练记录的“代码快照”（便于复现）
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
