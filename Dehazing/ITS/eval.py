"""
eval.py：去雾任务的测试（评估）流程，单张图片逐张推理并统计平均 PSNR / SSIM。
逻辑与 OTS 的 eval.py 基本一致，差异仅在于没有显式 torch.cuda.synchronize()
（计时略不精确但差别不大），以及 SSIM 打印格式为 %.5f。
主要步骤：
  1. 加载模型权重、eval 模式、关梯度。
  2. 输入 reflect 填充到 32 的整数倍后推理，再裁回原始尺寸。
  3. 取模型第 3 个输出（[2]，全分辨率）作为预测，计算 PSNR 与 SSIM。
"""
import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
from pytorch_msssim import ssim
import torch.nn.functional as f

from skimage import img_as_ubyte
import cv2

def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])   # 载入权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)  # 逐张测试
    torch.cuda.empty_cache()
    adder = Adder()            # 累计单张耗时
    model.eval()               # 推理模式
    factor = 32                # 对齐因子
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)   # 取 32 整数倍
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            tm = time.time()

            pred = model(input_img)[2]          # 全分辨率输出
            pred = pred[:,:,:h,:w]              # 裁掉填充
            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()


            label_img = (label_img).cuda()
            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))   # PSNR
            down_ratio = max(1, round(min(H, W) / 256))	 # 超长宽图先降采样再算 SSIM
            ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)	
            print('%d iter PSNR_dehazing: %.2f ssim: %f' % (iter_idx + 1, psnr_val, ssim_val))
            ssim_adder(ssim_val)

            if args.save_image:                # 可选：保存复原图
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

            psnr_mimo = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr_val)

            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr_mimo, elapsed))

        # 输出整批平均指标
        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.5f dB' % (ssim_adder.average()))

        print("Average time: %f" % adder.average())