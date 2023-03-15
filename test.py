import numpy
import torch
from numpy import double
import util.util as util
from options.train_options import TrainOptions
from models.models import create_model
from data.data_loader import *
from PIL import Image
import cv2
import os

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def fuse(opt, model, imgA_path, imgB_path, img_fused):
    imgA = cv2.imread(imgA_path)
    imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
    imgB = cv2.imread(imgB_path)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
    imgA = double(imgA) / 255
    imgB = double(imgB) / 255
    w = imgA.shape[0]
    h = imgA.shape[1]
    imgA = cv2.resize(imgA, (592, 400))
    imgB = cv2.resize(imgB, (592, 400))

    imgA = torch.from_numpy(imgA)
    imgB = torch.from_numpy(imgB)

    imgA = imgA.unsqueeze(0)
    imgB = imgB.unsqueeze(0)

    imgA = imgA.permute(0, 3, 2, 1).float()
    imgB = imgB.permute(0, 3, 2, 1).float()

    imgA = imgA.cuda()
    imgB = imgB.cuda()

    output1 = model.attentionnet.forward(imgA)
    output2 = model.attentionnet2.forward(imgB)
    output3 = output1 * imgA + output2 * imgB

    fake_B = output3
    fake_B_gray = torch.empty(1, 1, fake_B.shape[2], fake_B.shape[3])
    R = fake_B[0][0]
    G = fake_B[0][1]
    B = fake_B[0][2]
    fake_B_gray[0][0] = 0.299 * R + 0.587 * G + 0.114 * B

    if(opt.model == 'separate'):
        output3_1, refine_1 = model.netG_A.forward(
            fake_B, fake_B_gray.cuda())
        output3_2, refine_2 = model.netG_B.forward(
            fake_B, fake_B_gray.cuda())
        output3 = output3_1*0.6+output3_2*0.4

    output3 = util.tensor2im_2(output3.detach())
    output3 = cv2.resize(output3, (h, w))
    outputimage3 = Image.fromarray(numpy.uint8(output3))
    outputimage3.save(img_fused)


def testphotos(u_path, o_path, save_path):
    opt = TrainOptions().parse()
    model = create_model(opt)

    weight_path = r'./checkpoints/'
    model.load_network(model.attentionnet, 'A_1', weight_path)
    model.load_network(model.attentionnet2, 'A_2', weight_path)
    model.load_network(model.netG_A, 'G_A', weight_path)
    model.load_network(model.netG_B, 'G_B', weight_path)

    model.attentionnet.eval()
    model.attentionnet2.eval()
    model.netG_A.eval()
    model.netG_B.eval()

    img_names = sorted(os.listdir(u_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(img_names)):
        name = img_names[i]
        u = os.path.join(u_path, name)
        o = os.path.join(o_path, name)
        save = os.path.join(save_path, name)
        fuse(opt, model, u, o, save)
        print(save)
    #
    #
    # for i in range(1, 5):
    #     imgA_path = "./datasets/SICE/test1/trainA/" + \
    #         str(i).zfill(3)+".png"
    #     imgB_path = "./datasets/SICE/test1/trainB/" + \
    #         str(i).zfill(3)+".png"
    #     imgpath = weight_path + "fused_results_test100_2/"
    #     if (os.path.exists(imgA_path) == 0 or os.path.exists(imgB_path) == 0):
    #         continue
    #     if (os.path.exists(imgpath) == 0):
    #         os.makedirs(imgpath)
    #     img_fused2 = imgpath + '195' + "/"
    #     if(os.path.exists(img_fused2) == 0):
    #         os.makedirs(img_fused2)
    #     img_fused = img_fused2 + str(i)+".png"
    #     print(img_fused)
    #     fuse(opt, model, imgA_path, imgB_path, img_fused)


if __name__ == '__main__':

    u_path = r'datasets/SICE/under'
    o_path = r'datasets/SICE/over'
    save_path = r'result/SICE/'
    testphotos(u_path, o_path, save_path)

