import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class BasicSpatialAttenionNet(torch.nn.Module):
    def __init__(self):
        super(BasicSpatialAttenionNet, self).__init__()

        self.fe1 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.fe2 = torch.nn.Conv2d(64, 64, 3, 1, 1)

        self.sAtt_1 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = torch.nn.Conv2d(64 * 2, 64, 1, 1, bias=True)
        self.sAtt_3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.sAtt_4 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.sAtt_5 = torch.nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.sAtt_L1 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.sAtt_L2 = torch.nn.Conv2d(64 * 2, 64, 3, 1, 1, bias=True)
        self.sAtt_L3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, alignedframe):
        # feature extraction
        att = self.lrelu(self.fe1(alignedframe))
        att = self.lrelu(self.fe2(att))

        # spatial attention
        att = self.lrelu(self.sAtt_1(att))

        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att2 = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att2))

        att2 = F.interpolate(att_L,
                             size=[alignedframe.size(2), alignedframe.size(3)],
                             mode='bilinear', align_corners=False)
        att = att + att2
        att = self.sAtt_5(att)
        att = torch.sigmoid(att)

        return att


def define_basicatt1(gpu_ids):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    netatt1 = BasicSpatialAttenionNet()
    netatt1.apply(weights_init)
    netatt1.cuda()
    return netatt1


def define_basicatt2(gpu_ids):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    netatt2 = BasicSpatialAttenionNet()
    netatt2.apply(weights_init)
    netatt2.cuda()
    return netatt2


def define_G(opt, retina):
    netG = Unet_resize_conv(opt=opt, skip=1, retina=retina)
    netG.apply(weights_init)
    netG.cuda()
    return netG


class NoNormDiscriminator(nn.Module):
    def __init__(self):
        super(NoNormDiscriminator, self).__init__()
        self.gpu_ids = 0
        ndf = 64
        kw = 4
        n_layers = 3
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(3, 64, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def define_D():
    netD = NoNormDiscriminator()
    netD.apply(weights_init)
    netD.cuda()
    return netD


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, opt):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        if opt.vgg_choose != "no_maxpool":
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)

        if opt.vgg_choose != "no_maxpool":
            if opt.vgg_maxpooling:
                h = F.max_pool2d(h, kernel_size=2, stride=2)

        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        if opt.vgg_choose == "relu5_1":
            return relu5_1


def vgg_preprocess(batch, opt):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    return batch


class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        img_fea = vgg(img_vgg, self.opt)
        target_fea = vgg(target_vgg, self.opt)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


def load_vgg16(model_dir, gpu_ids):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    vgg = Vgg16()
    vgg.load_state_dict(torch.load('./checkpoints/vgg16.weight'))
    vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg


class Unet_resize_conv(nn.Module):
    def __init__(self, opt, skip, retina):
        super(Unet_resize_conv, self).__init__()
        self.opt = opt
        self.skip = skip
        self.is_retina = retina
        p = 1
        if opt.self_attention:

            self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
            self.downsample_1 = nn.MaxPool2d(2)
            self.downsample_2 = nn.MaxPool2d(2)
            self.downsample_3 = nn.MaxPool2d(2)
            self.downsample_4 = nn.MaxPool2d(2)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_2 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_2 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_2 = nn.BatchNorm2d(512)

        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_1 = nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_2 = nn.BatchNorm2d(256)

        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_2 = nn.BatchNorm2d(128)

        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_2 = nn.BatchNorm2d(64)

        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 3, 1)

    def depth_to_space(self, input, block_size):
        block_size_sq = block_size * block_size
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / block_size_sq)
        s_width = int(d_width * block_size)
        s_height = int(d_height * block_size)
        t_1 = output.resize(batch_size, d_height, d_width,
                            block_size_sq, s_depth)
        spl = t_1.split(block_size, 3)
        stack = [t_t.resize(batch_size, d_height, s_width, s_depth)
                 for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).resize(batch_size, s_height, s_width,
                                                                                     s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

    def forward(self, input, gray):
        flag = 0
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            gray = avg(gray)
            flag = 1

        if self.opt.self_attention:
            gray_2 = self.downsample_1(gray)
            gray_3 = self.downsample_2(gray_2)
            gray_4 = self.downsample_3(gray_3)
            gray_5 = self.downsample_4(gray_4)
        if self.opt.use_norm == 1:
            if self.opt.self_attention:
                x = self.bn1_1(self.LReLU1_1(
                    self.conv1_1(torch.cat((input, gray), 1))))
            conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
            x = self.max_pool1(conv1)

            x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
            conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
            x = self.max_pool2(conv2)

            x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
            conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
            x = self.max_pool3(conv3)

            x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
            conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
            x = self.max_pool4(conv4)

            x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
            x = x * gray_5 if self.opt.self_attention else x
            conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))

            conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear')
            conv4 = conv4 * gray_4 if self.opt.self_attention else conv4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
            x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
            conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

            conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
            conv3 = conv3 * gray_3 if self.opt.self_attention else conv3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
            x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
            conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

            conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
            conv2 = conv2 * gray_2 if self.opt.self_attention else conv2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
            x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
            conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

            conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
            conv1 = conv1 * gray if self.opt.self_attention else conv1
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
            conv9 = self.LReLU9_2(self.conv9_2(x))

            latent = self.conv10(conv9)

            if self.opt.times_residual:
                latent = latent * gray

            if self.skip:
                if self.is_retina == 1:
                    latent = F.sigmoid(latent)
                    output = input / (latent + 0.00001)
                else:
                    output = latent + input * self.opt.skip
            else:
                output = latent

        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
            gray = F.upsample(gray, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output
