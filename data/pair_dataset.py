import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import torch
from pdb import set_trace as st
import numpy as np
import cv2
import time


class PairDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')

        self.A_paths=[]
        self.B_paths=[]
        self.C_paths=[]
        for i in range(1, 259):
            self.A_paths.append(self.dir_A + "/" + str(i) + ".jpg")
            self.B_paths.append(self.dir_B + "/" + str(i) + ".jpg")
            self.C_paths.append(self.dir_C + "/" + str(i) + ".jpg")
        if(opt.fullinput==1):

            self.D_paths_set=[]
            for i in range(1, 259):

                dir_D = "smallSICE/" + str(i)
                tempset = make_dataset(dir_D)
                # tempset = sorted(tempset)
                self.D_paths_set.append(tempset)



        # self.A_paths = sorted(self.A_paths)
        # self.B_paths = sorted(self.B_paths)
        # # self.E_paths = sorted(self.E_paths)
        # self.C_paths = sorted(self.C_paths)




        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        # self.E_size = len(self.E_paths)
        self.C_size = len(self.C_paths)
        self.D_sizes = []
        if (self.opt.fullinput == 1):
            for i in range(0, 258):
                self.D_sizes.append(len(self.D_paths_set[i]))

        transform_list = []

        transform_list += [transforms.ToTensor()]
        # transform_list = [transforms.ToTensor()]

        self.transform1 = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
            transforms.ToTensor()
        ])

        self.transform = transforms.Compose(transform_list)
        # self.transform = get_transform(opt)

    def collate_fn(self,batchs):
        from torch.utils.data.dataloader import default_collate
        lanes = []
        if(self.opt.fullinput==1):
            for i in range(0,len(batchs['D'])):
                lanes.append(batchs['D'][i])
            lanes = default_collate(lanes)
            batchs['D'] = lanes
        return batchs
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        # E_path = self.B_paths[index % self.E_size]
        C_path = self.C_paths[index % self.C_size]
        if (self.opt.fullinput == 1):
            D_path=[]
            for i in range(0,len(self.D_paths_set[index % self.A_size])):
                temppath = self.D_paths_set[index % self.A_size][i]
                D_path.append(temppath)

        A_img = Image.open(A_path).convert('RGB')
        a = Image.open(A_path).convert('RGB')
        B_img = Image.open(A_path.replace("low", "high").replace("A", "B")).convert('RGB')
        b = Image.open(A_path.replace("low", "high").replace("A", "B")).convert('RGB')
        C_img = Image.open(A_path.replace("low", "gt").replace("A", "C")).convert('RGB')
        if (self.opt.fullinput == 1):
            D_img = []

            for i in range(0,len(D_path)):

                tempimage = Image.open(D_path[i])
                D_img.append(tempimage)


        A_img = self.transform(A_img)
        a = self.transform(a)
        B_img = self.transform(B_img)
        b = self.transform(b)
        aaa = C_img
        C_img = self.transform(C_img)
        if (self.opt.fullinput == 1):
            for i in range(0,len(D_img)):
                D_img[i] = self.transform(D_img[i])

        C_d = self.transform1(aaa)
        # E_img = self.transform(E_img)
        w = A_img.size(2)
        h = A_img.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A_img = A_img[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]
        a = a[:, h_offset:h_offset + self.opt.fineSize,
            w_offset:w_offset + self.opt.fineSize]
        B_img = B_img[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]
        b = b[:, h_offset:h_offset + self.opt.fineSize,
            w_offset:w_offset + self.opt.fineSize]
        C_img = C_img[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]
        # E_img = E_img[:, h_offset:h_offset + self.opt.fineSize,
        #         w_offset:w_offset + self.opt.fineSize]
        C_d = C_d[:, h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]
        if (self.opt.fullinput == 1):
            for i in range(0,len(D_img)):
                D_img[i] = D_img[i][:, h_offset:h_offset + self.opt.fineSize,
                    w_offset:w_offset + self.opt.fineSize]

        if self.opt.resize_or_crop == 'no':
            r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
            A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else:

            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                a = a.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
                b = b.index_select(2, idx)
                C_img = C_img.index_select(2, idx)
                C_d = C_d.index_select(2, idx)
                if (self.opt.fullinput == 1):
                    for i in range(0,len(D_img)):
                        D_img[i] = D_img[i].index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                a = a.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
                b = b.index_select(1, idx)
                C_img = C_img.index_select(1, idx)
                C_d = C_d.index_select(1, idx)
                if (self.opt.fullinput == 1):
                    for i in range(0,len(D_img)):
                        D_img[i] = D_img[i].index_select(1, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times, self.opt.high_times) / 100.
                input_img = (A_img + 1) / 2. / times
                input_img = input_img * 2 - 1
            else:
                input_img = A_img
            r, g, b = input_img[0] + 1, input_img[1] + 1, input_img[2] + 1
            A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
            A_gray = torch.unsqueeze(A_gray, 0)
        if (self.opt.fullinput == 1):
            batch= {'A': A_img, 'B': B_img, 'C': C_img, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path,
                    'C_gray': C_d,'D':D_img}
        else:
            batch = {'A': A_img, 'B': B_img, 'C': C_img, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path,
                     'C_gray': C_d}
        batch = self.collate_fn(batch)
        return batch

    def __len__(self):
        return self.A_size

    def name(self):
        return 'PairDataset'
