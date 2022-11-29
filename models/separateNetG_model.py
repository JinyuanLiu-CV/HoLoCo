import numpy
import numpy as np
import torch
from . import CR
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
import random
from . import networks
from MEFSSIM.lossfunction import MEFSSIM
import torch.nn.functional as F
class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, 3, size, size)
        self.input_B = self.Tensor(nb, 3, size, size)
        self.input_C = self.Tensor(nb, 3, size, size)
        if(opt.fullinput==1):
            self.tempinput_D = self.Tensor(nb, 3, size, size)
            self.input_D = []
        self.input_img = self.Tensor(nb, 3, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)
        self.input_C_gray = self.Tensor(nb, 1, size, size)

        if opt.vgg > 0:
            self.vgg_loss = networks.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        self.attentionnet = networks.define_basicatt1(gpu_ids=opt.gpu_ids)
        self.attentionnet2 = networks.define_basicatt2(gpu_ids=opt.gpu_ids)

        self.netG_A = networks.define_G(self.opt,retina=1)
        self.netG_A.cuda()

        self.netG_B = networks.define_G(self.opt,retina=0)
        self.netG_B.cuda()


        self.netD_A = networks.define_D()
        # if self.opt.patchD_3:
        #     self.netD_P = networks.define_D()
        self.attentionnet.train()
        self.attentionnet2.train()
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        # if self.opt.patchD_3:
        #     self.netD_P.train()

        self.old_lr = opt.lr
        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        self.optimizer_basic1 = torch.optim.Adam(self.attentionnet.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_basic2 = torch.optim.Adam(self.attentionnet2.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G2 = torch.optim.Adam(self.netG_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # if self.opt.patchD_3:
        #     self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A']
        input_B = input['B']
        input_C = input['C']
        if (self.opt.fullinput == 1):
            input_D_list = input['D']

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_C.resize_(input_C.size()).copy_(input_C)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if (self.opt.fullinput == 1):
            self.input_D=[]
            for i in range(0,input_D_list.shape[1]):
                temp = input_D_list[0][i]
                self.input_D.append(temp)

    def get_image_paths(self):
        return self.image_paths

    def backward_basic(self):
        self.MSE = torch.nn.MSELoss()
        self.loss_basic = self.MSE(self.output1, self.gt)
        self.loss_basic.backward()

    def backward_G(self):
        self.loss_DH=0
        self.loss_DH1=0
        self.loss_DH2=0
        self.loss_patchDH=0
        self.loss_patchDH1=0
        self.loss_patchDH2=0
        if(self.opt.contract_weight>0):
            self.CTLoss = CR.ContrastLoss()
            self.CTLoss.cuda()


            if(self.opt.fullinput==0):
                if(self.opt.hasglobal):
                    self.loss_DH1 = self.CTLoss(vgg=self.vgg, a=self.fake_C, p=self.gt, n=self.under, opt=self.opt,mode='single',group_n=0)
                    self.loss_DH2 = self.CTLoss(vgg=self.vgg, a=self.fake_C, p=self.gt, n=self.over, opt=self.opt,mode='single',group_n=0)
                if(self.opt.patchD_3>0):
                    self.loss_patchDH1 = self.CTLoss(vgg=self.vgg, a=self.fake_patch_1, p=self.real_patch_1, n=self.input_patch_1, opt=self.opt,mode='multiple',group_n=0)
                    self.loss_patchDH2 = self.CTLoss(vgg=self.vgg, a=self.fake_patch_1, p=self.real_patch_1, n=self.input_patch_2, opt=self.opt,mode='multiple',group_n=0)
            else:
                if (self.opt.hasglobal):
                    self.loss_DH = self.CTLoss(vgg=self.vgg, a=self.fake_C, p=self.gt, n=self.full, opt=self.opt,
                                                mode='single',group_n=1)
                if (self.opt.patchD_3 > 0):
                    self.loss_patchDH = self.CTLoss(vgg=self.vgg, a=self.fake_patch_1, p=self.real_patch_1,
                                                     n=self.full_patch, opt=self.opt, mode='multiple',group_n=1)
        self.loss_G_A=0
        if(self.opt.gan_weight>0):
            pred_fake = self.netD_A.forward(self.fake_C)
            # if self.opt.patchD_3 > 0:
            #     pred_fake_patch=[]
            #     for i in range(0, self.opt.patchD_3):
            #         pred_fake_patch.append(self.netD_A.forward(self.fake_patch_1[i]))
            self.loss_G_A = self.criterionGAN(pred_fake, True)
            # if self.opt.patchD_3>0:
            #     self.loss_G_A_patch=0
            #     for i in range(0,self.opt.patchD_3):
            #         self.loss_G_A_patch+= self.criterionGAN(pred_fake_patch[i],True)
            #     self.loss_G_A = self.loss_G_A + self.loss_G_A_patch

            loss_G_A = 0
            if not self.opt.D_P_times2:
                self.loss_G_A = self.loss_G_A + loss_G_A
            else:
                self.loss_G_A += loss_G_A * 2
        self.loss_vgg_b=0
        if self.opt.vgg_weight>0:
            if self.opt.vgg > 0:
                 self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg,
                                                                  self.fake_C,
                                                                  self.gt) * self.opt.vgg if self.opt.vgg > 0 else 0
        self.mse_loss=0
        if self.opt.mse_weight>0:
            loss_mse = nn.MSELoss()
            self.mse_loss=loss_mse(self.fake_C,self.gt)

        if self.opt.ssim_loss>0:
            batchsize,rows, columns, channels = self.input_A.shape
            imgset = numpy.ones([2, 3, self.input_A.shape[2], self.input_A.shape[3]])
            myMEFSSIM = MEFSSIM()
            self.ssimscore=0
            fakeC = torch.permute(self.fake_C, [0, 2, 3, 1])
            for i in range(0,batchsize):
                imgset[0,:, :, :] = self.input_A[i].cpu().numpy()
                imgset[1,:, :, :] = self.input_B[i].cpu().numpy()
                imgset_tensor = torch.tensor(imgset)
                imgset_tensor = torch.permute(imgset_tensor,[0,2,3,1])

                ssimresult = myMEFSSIM.forward(fakeC[i].unsqueeze(0),imgset_tensor.cuda().float())
                if(np.isnan(ssimresult.item())==0):
                    self.ssimscore = self.ssimscore + ssimresult



        mw= self.opt.mse_weight
        vw= self.opt.vgg_weight
        gw= self.opt.gan_weight
        cw= self.opt.contract_weight
        glr=self.opt.global_local_rate
        if(self.opt.fullinput==0):


            if(self.opt.hasglobal==1 and self.opt.patchD_3==0):
                self.loss_G = mw* self.mse_loss+ gw*self.loss_G_A +vw*self.loss_vgg_b + cw/2* self.loss_DH1 +cw/2*self.loss_DH2
            elif(self.opt.hasglobal==0 and self.opt.patchD_3>0):
                self.loss_G = mw* self.mse_loss+ gw*self.loss_G_A +vw*self.loss_vgg_b + cw/2* self.loss_patchDH1 +cw/2*self.loss_patchDH2
            else:
                self.loss_G = mw* self.mse_loss+ gw*self.loss_G_A +vw*self.loss_vgg_b + cw/2*(glr/(glr+1))* self.loss_DH1 +cw/2*(glr/(glr+1))*self.loss_DH2 +cw/2*(1/(glr+1))*self.loss_patchDH1+cw/2*(1/(glr+1))*self.loss_patchDH2
        else:
            if (self.opt.hasglobal == 1 and self.opt.patchD_3 == 0):
                self.loss_G = mw* self.mse_loss+ gw * self.loss_G_A + vw * self.loss_vgg_b + cw * self.loss_DH
            elif (self.opt.hasglobal == 0 and self.opt.patchD_3 > 0):
                self.loss_G = mw* self.mse_loss + gw * self.loss_G_A + vw * self.loss_vgg_b + cw * self.loss_patchDH
            else:
                self.loss_G = mw* self.mse_loss+ gw * self.loss_G_A + vw * self.loss_vgg_b + cw*(glr/(glr+1)) * self.loss_DH + cw*(1/(glr+1)) * self.loss_patchDH
        if(self.opt.ssim_loss):
            self.loss_G = self.loss_G + self.opt.ssim_loss * (self.opt.batchSize-self.ssimscore)
        #self.loss_G= 0.1*self.loss_G_A+0.9*self.loss_vgg_b
        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        if self.opt.patchD_3>0:
            pred_fake_patch=[]
            pred_real_patch=[]
            for i in range(0,self.opt.patchD_3):
                pred_real_patch.append(netD.forward(self.real_patch_1[i].detach()))
                pred_fake_patch.append(netD.forward(self.fake_patch_1[i].detach()))

        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())
        # if self.opt.use_wgan:
        #     loss_D_real = pred_real.mean()
        #     loss_D_fake = pred_fake.mean()
        #     loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD,
        #                                                                                  real.data, fake.data)
        # if self.opt.use_ragan and use_ragan:
        glr = self.opt.global_local_rate
        # if self.opt.patchD_3>0:
        #     loss_D_patch=0
        #     for i in range(0,self.opt.patchD_3):
        #         loss_D_patch+=(self.criterionGAN(pred_real_patch[i] - torch.mean(pred_fake_patch[i]), True) +
        #               self.criterionGAN(pred_fake_patch[i] - torch.mean(pred_real_patch[i]), False)) / 2
        #     loss_D = ((self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
        #               self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2)*glr/(glr+1)+loss_D_patch*1/(glr+1)
        # else:
        loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2

        # else:
        #     loss_D_real = self.criterionGAN(pred_real, True)
        #     loss_D_fake = self.criterionGAN(pred_fake, False)
        #     loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_C = self.fake_C
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.gt, fake_C, True)
        self.loss_D_A.backward()

    def forward1(self):
        self.under = Variable(self.input_A)
        self.over = Variable(self.input_B)
        self.gt = Variable(self.input_C)
        self.full= []
        if(self.opt.fullinput):
            for i in range(len(self.input_D)):
                temp = Variable(self.input_D[i])
                self.full.append(temp.unsqueeze(0))
        self.gt_gray = Variable(self.input_C_gray)
        self.a1 = self.attentionnet.forward(self.under)
        self.a2 = self.attentionnet2.forward(self.over)
        self.output1 = self.a1 * self.under + self.a2 * self.over
        #self.fake_B = self.Dehazenet.forward(self.output1)

    def forward2(self,netG):
        self.fake_B = self.output1.detach()
        if (self.opt.batchSize == 1):
            self.fake_B_gray = torch.empty(1, 1, self.fake_B.shape[2], self.fake_B.shape[3])
            R = self.fake_B[0][0]
            G = self.fake_B[0][1]
            B = self.fake_B[0][2]
            self.fake_B_gray[0][0] = 0.299 * R + 0.587 * G + 0.114 * B
        elif (self.opt.batchSize == 2):
            self.fake_B_gray = torch.empty(2, 1, self.fake_B.shape[2], self.fake_B.shape[3])
            R = self.fake_B[0][0]
            G = self.fake_B[0][1]
            B = self.fake_B[0][2]
            self.fake_B_gray[0][0] = 0.299 * R + 0.587 * G + 0.114 * B
            R = self.fake_B[1][0]
            G = self.fake_B[1][1]
            B = self.fake_B[1][2]
            self.fake_B_gray[1][0] = 0.299 * R + 0.587 * G + 0.114 * B

        if self.opt.skip == 1:
            if(netG=='netG_A'):
            # self.fake_C, self.latent_fake_A = self.netG_A.forward(self.fake_B,self.fake_B_gray)
                self.fake_C, self.latent_fake_A1 = self.netG_A.forward(self.fake_B, self.fake_B_gray.cuda())
                self.fake_C0 = self.fake_C
            elif(netG == 'netG_B'):
                self.fake_C, self.latent_fake_A2 = self.netG_B.forward(self.fake_B, self.fake_B_gray.cuda())


        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            self.input_patch_2 = []
            self.full_patch = []

            w = self.fake_C.size(3)
            h = self.fake_C.size(2)

            for i in range(self.opt.patchD_3):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.fake_patch_1.append(self.fake_C[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                         w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_patch_1.append(self.gt[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                         w_offset_1:w_offset_1 + self.opt.patchSize])
                self.input_patch_1.append(self.under[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                          w_offset_1:w_offset_1 + self.opt.patchSize])

                self.input_patch_2.append(self.over[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                          w_offset_1:w_offset_1 + self.opt.patchSize])
                temppatches = []
                for j in range(len(self.full)):
                    temppatches.append(self.full[j][:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                       w_offset_1:w_offset_1 + self.opt.patchSize])
                self.full_patch.append(temppatches)

    def optimize_parameters(self, epoch):

        self.forward1()

        self.optimizer_basic1.zero_grad()
        self.optimizer_basic2.zero_grad()
        self.backward_basic()
        self.optimizer_basic1.step()
        self.optimizer_basic2.step()

        self.forward2(netG='netG_A')
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.forward2(netG='netG_B')
        self.optimizer_G2.zero_grad()
        self.backward_G()
        self.optimizer_G2.step()

        #self.optimizer_DH.step()



        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()

    def get_current_errors(self, epoch):
        G_A = self.loss_G_A
        MSE = self.mse_loss
        if(self.opt.fullinput==0):
            if(self.opt.hasglobal):
                DH1=self.loss_DH1
                DH2=self.loss_DH2
            if(self.opt.patchD_3):
                DHp1 = self.loss_patchDH1
                DHp2 = self.loss_patchDH2

        else:
            if (self.opt.hasglobal):
                DH = self.loss_DH
            if (self.opt.patchD_3):
                DHp = self.loss_patchDH
        vgg=0
        if self.opt.vgg > 0:
            if self.opt.vgg_weight>0:
                vgg = self.loss_vgg_b.item() / self.opt.vgg if self.opt.vgg > 0 else 0
            mefssim = 0
            if self.opt.ssim_loss>0:
                mefssim = self.opt.batchSize-self.ssimscore
            if(self.opt.fullinput==1):
                if(self.opt.hasglobal==1 and self.opt.patchD_3==0):
                    return OrderedDict([('MSE',MSE),('G_A', G_A), ("vgg", vgg),("DH",DH),('MEFSSIM',mefssim)])
                if (self.opt.hasglobal == 0 and self.opt.patchD_3 >0 ):
                    return OrderedDict([('MSE',MSE),('G_A', G_A), ("vgg", vgg), ("DHpatch", DHp),('MEFSSIM',mefssim)])
                if (self.opt.hasglobal == 1 and self.opt.patchD_3 >0 ):
                    return OrderedDict([('MSE',MSE),('G_A', G_A), ("vgg", vgg), ("DH", DH), ("DHp",DHp),('MEFSSIM',mefssim)])
            else:
                if (self.opt.hasglobal == 1 and self.opt.patchD_3 == 0):
                    return OrderedDict([('MSE',MSE),('G_A', G_A), ("vgg", vgg), ("DH1", DH1),("DH2",DH2),('MEFSSIM',mefssim)])
                if (self.opt.hasglobal == 0 and self.opt.patchD_3 > 0):
                    return OrderedDict([('MSE',MSE),('G_A', G_A), ("vgg", vgg), ("DHpatch1", DHp1), ("DHpatch2", DHp2),('MEFSSIM',mefssim)])
                if (self.opt.hasglobal == 1 and self.opt.patchD_3 > 0):
                    return OrderedDict([('MSE',MSE),('G_A', G_A), ("vgg", vgg), ("DH1", DH1), ("DH2", DH2), ("DHpatch1", DHp1),
                                        ("DHpatch2", DHp2),('MEFSSIM',mefssim)])

    def get_current_visuals(self):
        under = util.tensor2im(self.under.data)
        over = util.tensor2im(self.over.data)
        gt = util.tensor2im(self.gt.data)
        a1 = util.tensor2im(self.a1.detach())
        a2 = util.tensor2im(self.a2.detach())
        fake_A = util.tensor2im(self.output1.data)
        fake_C = util.tensor2im(self.fake_C.data)
        fake_C0 = util.tensor2im(self.fake_C0.data)
        if self.opt.skip > 0:
            latent_fake_A1 = util.tensor2im(self.latent_fake_A1.data)
            latent_fake_A2 = util.tensor2im(self.latent_fake_A2.data)
            return OrderedDict(
                [('under', under), ('over', over), ('gt', gt), ('a1', a1), ('a2', a2), ('fake_A', fake_A),('fake_C', fake_C),
                 #('fake_B', fake_B),
                 ('fake_C1', fake_C0),
                 ('fake_C2', fake_C),
                 ('latent_fake_A1', latent_fake_A1),
                 ('latent_fake_A2', latent_fake_A2),
                 ])

    def save(self, label):
        self.save_network(self.attentionnet, 'A_1', label, self.gpu_ids)
        self.save_network(self.attentionnet2, 'A_2', label, self.gpu_ids)
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        #self.save_network(self.netD_P, 'D_P', label, self.gpu_ids)

    def update_learning_rate(self):

        if self.opt.new_lr:
            lr = self.old_lr / 2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
