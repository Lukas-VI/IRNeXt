"""
valid.py：训练过程中的验证流程（OTS 数据集，在 test 子集上计算平均 PSNR）。
与 ITS/GoPro 的 valid.py 完全同构，仅验证集变量名用 ots。
用于训练中周期性评估模型，返回平均 PSNR 供 train.py 判断是否保存 Best 模型。
"""
import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ots = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)  # 验证集(用 test 子集)
    model.eval()     # 推理模式
    psnr_adder = Adder()

    with torch.no_grad():     # 关梯度
        print('Start Evaluation')
        factor = 32           # 对齐因子
        for idx, data in enumerate(ots):
            input_img, label_img = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)   # 取 32 整数倍
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):   # 建按 epoch 命名的结果目录
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img)[2]     # 全分辨率输出
            pred = pred[:,:,:h,:w]         # 裁掉填充

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)   # 单张 PSNR

            psnr_adder(psnr)
            print('\r%03d'%idx, end=' ')   # 进度显示

    print('\n')
    model.train()     # 验证完切回训练模式
    return psnr_adder.average()   # 返回平均 PSNR