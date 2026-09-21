"""
eval.py：去雾任务的测试（评估）流程，单张图片逐张推理并统计平均 PSNR / SSIM。
主要步骤：
  1. 加载训练好的模型权重（--test_model）；模型设为 eval 模式；关梯度。
  2. 为保证输入尺寸能被网络整除（网络含多次 stride=2 与特征拼接），先把每张输入用
     reflect 方式扩展到 factor=32 的整数倍，推理完再裁回原始尺寸。
  3. 用模型第 3 个输出（[2]，即全分辨率那一张）作为最终预测。
  4. 计算 PSNR 和 SSIM（SSIM 在按需下采样后计算，以处理超长宽图）；可选保存复原图。
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
# ---------------------------------------------------

def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])   # 载入权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)  # 逐张测试
    torch.cuda.empty_cache()   # 清空 GPU 缓存
    adder = Adder()            # 累计单张推理耗时
    model.eval()               # 切到推理模式(关闭 dropout/BN 更新)
    factor = 32                # 对齐网络的整除因子
    with torch.no_grad():      # 不计算梯度，节省显存与加速
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)   # 向上取整到 32 倍数
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')       # reflect 填充

            torch.cuda.synchronize()   # 等待 GPU 完成前面计算，确保计时准确
            tm = time.time()

            pred = model(input_img)[2]        # 取全分辨率输出
            pred = pred[:,:,:h,:w]            # 裁掉填充部分
            torch.cuda.synchronize()

            elapsed = time.time() - tm
            adder(elapsed)                    # 累加单张耗时

            pred_clip = torch.clamp(pred, 0, 1)   # 截断到 [0,1]

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()   # 转为 numpy 供 skimage 计算 PSNR
            label_numpy = label_img.squeeze(0).cpu().numpy()

            label_img = (label_img).cuda()
            # 用 GPU 上的 MSE 计算 PSNR(峰值信噪比)
            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))
            down_ratio = max(1, round(min(H, W) / 256))	 # 对超长宽图先降采样再算 SSIM 以省显存
            ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)	
            print('%d iter PSNR_dehazing: %.2f ssim: %f' % (iter_idx + 1, psnr_val, ssim_val))
            ssim_adder(ssim_val)

            if args.save_image:               # 可选：把复原图保存成图片
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
        print('The average SSIM is %.4f dB' % (ssim_adder.average()))

        print("Average time: %f" % adder.average())