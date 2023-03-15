import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models
from models.networks import PerceptualLoss

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.weights = [1.0, 1.0, 1.0, 1.0]
        self.ab = ablation

    def forward(self, vgg, a, p, n, opt, mode, group_n):
        loss = 0
        self.perceptualloss = PerceptualLoss(opt)
        if group_n == 0:
            if(mode == 'single'):
                for i in range(len(a)):
                    d_ap = self.perceptualloss.compute_vgg_loss(vgg, a, p)
                    if not self.ab:
                        d_an = self.perceptualloss.compute_vgg_loss(vgg, a, n)
                        contrastive = d_ap / (d_an + 1e-7)
                    else:
                        contrastive = d_ap

                    loss += self.weights[i] * contrastive
            elif(mode == 'multiple'):
                for i in range(len(a)):

                    d_ap = self.perceptualloss.compute_vgg_loss(
                        vgg, a[i], p[i])
                    if not self.ab:
                        d_an = self.perceptualloss.compute_vgg_loss(
                            vgg, a[i], n[i])
                        contrastive = d_ap / (d_an + 1e-7)
                    else:
                        contrastive = d_ap

                    loss += self.weights[i] * contrastive
            return loss
        