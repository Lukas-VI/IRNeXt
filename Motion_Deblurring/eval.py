"""
eval.py：运动去模糊任务的测试（评估）流程（GoPro）。
与去雾版相比做了简化：只计算并统计平均 PSNR（不算 SSIM）。
流程：载权重 → 输入 reflect 填充到 32 整数倍 → 取全分辨率输出裁回 → 算 PSNR → 求平均。
"""
import os
import torch
from torchvision.transforms import functional as F
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
import torch.nn.functional as f

factor = 32   # 对齐因子（模块级变量）

def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])   # 载权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)  # 逐张测试
    adder = Adder()            # 累计单张耗时
    model.eval()               # 推理模式

    with torch.no_grad():
        psnr_adder = Adder()
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

            if args.save_image:                # 可选：保存复原图
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)
                
            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)   # 单张 PSNR
            psnr_adder(psnr)
            print('%d iter PSNR: %.4f time: %f' % (iter_idx + 1, psnr, elapsed))

        # 输出整批平均 PSNR 与平均耗时
        print('==========================================================')
        print('The average PSNR is %.4f dB' % (psnr_adder.average()))
        print("Average time: %f" % adder.average())
