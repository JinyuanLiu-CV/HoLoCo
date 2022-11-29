    # -*- coding: utf-8 -*-
# @Time    : 2020/4/2 17:03
# @Author  : ZXF
# @File    : lossfunction.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


#-----------------------------------
# PSNR loss
# max = 1
class PsnrLoss(torch.nn.Module):
    def __init__(self):
        super(PsnrLoss, self).__init__()

    def forward(self, input, label):
        mse = torch.mean(torch.pow((input - label), 2))
        psnr = 10 * torch.log10(1 / mse)

        return 1.0/psnr


#-----------------------------------
# SSIM loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

#----------------------------------------------------------
# L1 and SSIM loss
class L1andSSIM(torch.nn.Module):
    def __init__(self, lambda1=1, lambda2=1):
        super(L1andSSIM, self).__init__()
        self.l_1 = lambda1
        self.l_2 = lambda2
        self.l1_loss = torch.nn.L1Loss()
        self.ssim_loss = SSIM()
    def forward(self, input, target):
        return self.l_1*self.l1_loss(input, target) - self.l_2*self.ssim_loss(input, target)

#-----------------------------------------------------------
# PSNR and SSIM loss
class PsnrandSSIM(torch.nn.Module):
    def __init__(self, lambda1=1, lambda2=1):
        super(PsnrandSSIM, self).__init__()
        self.l_1 = lambda1
        self.l_2 = lambda2
        self.psnr_loss = PsnrLoss()
        self.ssim_loss = SSIM()
    def forward(self, input, target):
        return - self.l_1*self.psnr_loss(input, target) - self.l_2*self.ssim_loss(input, target)

#-----------------------------------------------------------
# PSNR and L1 loss
class PsnrandL1(torch.nn.Module):
    def __init__(self, lambda1=1, lambda2=1):
        super(PsnrandL1, self).__init__()
        self.l_1 = lambda1
        self.l_2 = lambda2
        self.psnr_loss = PsnrLoss()
        self.l1_loss = torch.nn.L1Loss()
    def forward(self, input, target):
        return -self.l_1*self.psnr_loss(input, target) + self.l_2*self.l1_loss(input, target)