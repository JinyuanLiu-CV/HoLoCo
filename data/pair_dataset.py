import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from PIL import Image
import random
import torch


class PairDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')

        self.A_paths = []
        self.B_paths = []
        self.C_paths = []
        for i in range(1, 4):
            self.A_paths.append(self.dir_A + "/" + str(i).zfill(3) + ".png")
            self.B_paths.append(self.dir_B + "/" + str(i).zfill(3) + ".png")
            self.C_paths.append(self.dir_C + "/" + str(i).zfill(3) + ".png")

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)
        self.D_sizes = []

        transform_list = []

        transform_list += [transforms.ToTensor()]

        self.transform1 = transforms.Compose([
            # 彩色图像转灰度图像num_output_channels默认1
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        self.transform = transforms.Compose(transform_list)


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        C_path = self.C_paths[index % self.C_size]

        A_img = Image.open(A_path).convert('RGB')
        a = Image.open(A_path).convert('RGB')
        B_img = Image.open(A_path.replace(
            "low", "high").replace("A", "B")).convert('RGB')
        b = Image.open(A_path.replace(
            "low", "high").replace("A", "B")).convert('RGB')
        C_img = Image.open(A_path.replace(
            "low", "gt").replace("A", "C")).convert('RGB')

        A_img = self.transform(A_img)
        a = self.transform(a)
        B_img = self.transform(B_img)
        b = self.transform(b)
        aaa = C_img
        C_img = self.transform(C_img)

        C_d = self.transform1(aaa)
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
        C_d = C_d[:, h_offset:h_offset + self.opt.fineSize,
                  w_offset:w_offset + self.opt.fineSize]
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A_img = A_img.index_select(2, idx)
            a = a.index_select(2, idx)
            B_img = B_img.index_select(2, idx)
            b = b.index_select(2, idx)
            C_img = C_img.index_select(2, idx)
            C_d = C_d.index_select(2, idx)
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A_img = A_img.index_select(1, idx)
            a = a.index_select(1, idx)
            B_img = B_img.index_select(1, idx)
            b = b.index_select(1, idx)
            C_img = C_img.index_select(1, idx)
            C_d = C_d.index_select(1, idx)
        if (not self.opt.no_flip) and random.random() < 0.5:
            times = random.randint(
                self.opt.low_times, self.opt.high_times) / 100.
            input_img = (A_img + 1) / 2. / times
            input_img = input_img * 2 - 1
        else:
            input_img = A_img
        r, g, b = input_img[0] + 1, input_img[1] + 1, input_img[2] + 1
        A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        A_gray = torch.unsqueeze(A_gray, 0)
        batch = {'A': A_img, 'B': B_img, 'C': C_img, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path,
                    'C_gray': C_d}
        return batch

    def __len__(self):
        return self.A_size

    def name(self):
        return 'PairDataset'
