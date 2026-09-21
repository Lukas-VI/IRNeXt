"""
train.py：去雾任务（ITS 数据集）的训练主循环。
核心机制：
  1. 优化器 adam；学习率用 warmup(前 3 个 epoch 线性从 0 涨到目标) + 余弦退火
     两者串联（灵活用第三方 warmup_scheduler 包与 torch 内置 CosineAnnealingLR）。
  2. 损失函数 = 内容损失(L1，三尺度多尺度像素损失) + 0.1 * FFT 频域损失(L1 计算频谱实/虚部)。
     - 内容损失：模型输出三个尺度预测 pred_img[0/1/2]，分别与下采样成 1/4、1/2、1/1 的
       label(bool真值) 做 L1，再加权求和，形成深层监督，让每个分辨率都被约束。
     - 频域损失：对每个尺度的预测与对应真值做 FFT，比较频谱实部与虚部，迫使网络对齐频域信息，
       有助于恢复清晰的纹理/边缘（高频成分）。
  3. 每个 epoch 结束根据 valid_freq 在验证集上算平均 PSNR，并保存最优/Best 模型。
"""
import os
import torch
from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn

from warmup_scheduler import GradualWarmupScheduler

def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()  # 内容/像素损失与频域损失都用 L1 (MAE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)          # 一个 epoch 的迭代(批次)数
    warmup_epochs=3                     # 前 3 个 epoch 做学习率热身(warmup)
    # 余弦退火：在前 num_epoch-warmup_epochs 个 epoch 内把 lr 从峰值余弦式降到 eta_min
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-6)
    # 把余弦退火包在 warmup 调度器后面：前几轮线性升温后，交给余弦退火继续调度
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()    # 让调度器步进到初始状态（PyTorch warmup 库的习惯）
    epoch = 1
    if args.resume:  # 若给了 --resume，则从保存的 checkpoint 恢复模型/优化器/起始epoch
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter()            # tensorboard 记录损失/PSNR
    epoch_pixel_adder = Adder()         # 累计整个 epoch 的像素损失
    epoch_fft_adder = Adder()           # 累计整个 epoch 的频域损失
    iter_pixel_adder = Adder()          # 累计一个打印周期的像素损失
    iter_fft_adder = Adder()            # 累计一个打印周期的频域损失
    epoch_timer = Timer('m')            # 统计每个 epoch 耗时(分钟)
    iter_timer = Timer('m')             # 统计每个打印周期的耗时(分钟)
    best_psnr=-1                        # 记录迄今最佳验证 PSNR，用于保存 Best 模型

    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data   # input: 有雾图, label: 清晰真值
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)   # 三尺度预测 [1/4尺度, 1/2尺度, 全尺度]
            # 将真值下采样到 1/2 与 1/4，与模型各尺度预测对齐
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            l1 = criterion(pred_img[0], label_img4)   # 较小尺度(1/4)与预测[0]比较
            l2 = criterion(pred_img[1], label_img2)   # 中尺度(1/2)与预测[1]比较
            l3 = criterion(pred_img[2], label_img)    # 全尺度与原真值比较
            loss_content = l1+l2+l3                   # 内容(像素)损失

            # ---- 频域损失：对每个尺度的预测与真值做 FFT，比较频谱实部+虚部 ----
            label_fft1 = torch.fft.fft2(label_img4, dim=(-2,-1))
            label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

            pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2,-1))
            pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

            label_fft2 = torch.fft.fft2(label_img2, dim=(-2,-1))
            label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

            pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2,-1))
            pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

            label_fft3 = torch.fft.fft2(label_img, dim=(-2,-1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

            pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2,-1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3          # 三尺度频域损失求和

            loss = loss_content + 0.1 * loss_fft   # 总损失：频域项权重 0.1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)   # 梯度裁剪防梯度爆炸
            optimizer.step()

            iter_pixel_adder(loss_content.item())   # 累加当前 iter 内容损失
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())  # 累加当前 iter 到 epoch 总计
            epoch_fft_adder(loss_fft.item())

            if (iter_idx + 1) % args.print_freq == 0:  # 周期打印进度与损失
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, scheduler.get_lr()[0], iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)  # 记录到 tensorboard
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                
                iter_timer.tic()          # 重置打印周期计时与累加器
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
        # 每个 epoch 结束都覆盖保存一份“当前”模型（含优化器状态与epoch，用于断点续训）
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:   # 按 save_freq 周期性保存模型快照
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()             # 重置 epoch 累加器
        epoch_pixel_adder.reset()
        scheduler.step()                    # 学习率调度器步进到下一轮
        if epoch_idx % args.valid_freq == 0:  # 周期性在验证集评估 PSNR，保存最优模型
            val = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')  # 训练结束时保存最终模型
    torch.save({'model': model.state_dict()}, save_name)
