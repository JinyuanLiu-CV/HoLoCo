from asyncio.windows_events import NULL
import time
import numpy
import torch
import sys
import warnings
warnings.filterwarnings('ignore')
from numpy import double
from torchvision import transforms
# from models import networks, data_loader
# from models.lossfunction import PsnrLoss, SSIM
import util.util as util
from models import networks
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from data.data_loader import *
from util.visualizer import Visualizer
from PIL import Image
import cv2
import time
import os
import matplotlib.pyplot as plt
from numpy.core.defchararray import zfill



def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

def fuse(epoch,opt,model,imgA_path,imgB_path,img_fused):
    torch.cuda.synchronize()
    time0 = time.time()
    imgA = cv2.imread(imgA_path)
    imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
    imgB = cv2.imread(imgB_path)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
    imgA = double(imgA) / 255
    imgB = double(imgB) / 255
    w = imgA.shape[0]
    h = imgA.shape[1]
    imgA = cv2.resize(imgA, (592, 400))
    imgB = cv2.resize(imgB,(592,400))

    imgA = torch.from_numpy(imgA)
    imgB = torch.from_numpy(imgB)

    imgA = imgA.unsqueeze(0)
    imgB = imgB.unsqueeze(0)

    imgA = imgA.permute(0, 3, 2, 1).float()
    imgB = imgB.permute(0, 3, 2, 1).float()

    imgA=imgA.cuda()
    imgB=imgB.cuda()

    output1 = model.attentionnet.forward(imgA)
    output2 = model.attentionnet2.forward(imgB)
    output3 = output1 * imgA + output2 * imgB

    fake_B = output3
    fake_B_gray = torch.empty(1, 1, fake_B.shape[2], fake_B.shape[3])
    R = fake_B[0][0]
    G = fake_B[0][1]
    B = fake_B[0][2]
    fake_B_gray[0][0] = 0.299 * R + 0.587 * G + 0.114 * B

    output3_1, _ = model.netG_A.forward(fake_B, fake_B_gray.cuda())
    output3_2, _ = model.netG_B.forward(fake_B, fake_B_gray.cuda())
    output3 = output3_1/2+ output3_2/2

    torch.cuda.synchronize()
    time1 = time.time()
    print(time1-time0)

    output3 = util.tensor2im_2(output3.detach())
    output3 = cv2.resize(output3,(h,w))
    outputimage3 = Image.fromarray(numpy.uint8(output3))
    outputimage3.save(img_fused)

def testphotos(epoch,weight_path):
    opt = TrainOptions().parse()
    model = create_model(opt)
    model.load_network(model.attentionnet, 'A_1',epoch,weight_path)
    model.load_network(model.attentionnet2, 'A_2',epoch,weight_path)
    model.load_network(model.netG_A, 'G_A',epoch,weight_path)
    model.load_network(model.netG_B, 'G_B',epoch,weight_path)
    model.attentionnet.eval()
    model.attentionnet2.eval()
    model.netG_A.eval()
    model.netG_B.eval()
    path1 = './datasets/over/'
    path2 = './datasets/under/'
    names = sorted(os.listdir(path1))
    for i in names:
        imgA_path = path1 + i
        imgB_path = path2 + i
        save_path = "./results/"
        if (os.path.exists(imgA_path) == 0 or os.path.exists(imgB_path)==0):
            continue
        if (os.path.exists(save_path) == 0):
            os.makedirs(save_path)
        img_fused = save_path + i
        fuse(epoch,opt,model,imgA_path,imgB_path,img_fused)


if __name__ == '__main__':
    testphotos(195,"./checkpoints/")

