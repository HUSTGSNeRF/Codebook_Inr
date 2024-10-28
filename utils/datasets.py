import torch
from torch import nn
import numpy as np
import math
import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import scipy.io as scio
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop

from utils.modules import PositionalEncoding, Net

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import math

class codebook_dataset_mat(Dataset):
    # 文件路径已经写死，请勿随意修改文件路径
    def __init__(self, data_dir, PE=None):
        self.codebook_list = []  # 存放码本的list
        self.coordinate_list = np.array([[0.0, 0.0, 0.0]])  # 存放码本对应的归一化坐标
        # self.norm_para = []  # 存放坐标归一化参数  # 暂时先不归一化，看看效果
        self.root_path = data_dir  # 存放数据的根目录

        # 标准三维坐标：x的范围1~5、y的范围1~10、z的范围0~18
        # 文件的命名规则：“x-y-z”

        for x in range(1, 6):
            for y in range(1, 11):
                for z in range(19):
                    self.file_path = self.root_path + '/' + str(x) + '-' + str(y) + '-' + str(z) + '.mat'

                    if os.path.exists(self.file_path):  # 判断文件是否存在
                        self.codebook = scio.loadmat(self.file_path)['bits_max']
                        # self.codebook = self.codebook.reshape(-1)  # 直接在制作数据集时，把码本展开成一维向量
                        # 这里是CNN而不是DNN，所以不是展开成一维向量而是转成(通道数量，高度，宽度)
                        self.codebook = self.codebook.reshape((1, -1))  # 一定得指定channel的维数
                        self.codebook = self.codebook.astype(float)
                        self.codebook = self.codebook*2 - 1
                        self.codebook_list.append(self.codebook)

                        self.temp_coordinate = np.array([[0.1 * float(x), 0.1 * float(y), 0.1 * float(z)]])
                        self.coordinate_list = np.append(self.coordinate_list, self.temp_coordinate, axis=0)

        self.coordinate_list = np.delete(self.coordinate_list, 0, axis=0)
        self.coordinate_list = torch.from_numpy(self.coordinate_list).float()
        self.codebook_list = torch.from_numpy(np.array(self.codebook_list)).squeeze().float()
        
        if PE:
            print(PE)
            self.coordinate_list = PE(self.coordinate_list).view(len(self.coordinate_list), -1)
    def __getitem__(self, index):
        return self.coordinate_list[index, :], self.codebook_list[index]

    def __len__(self):
        return len(self.coordinate_list)

    def get_coordinate_list(self):
        return self.coordinate_list


class CodeBookDataset(Dataset):
    # np
    def __init__(self, coordinate, codebook, PE=None, vary_label=None, to_cuda=True,to11 = True):
        super().__init__()
        len_data = codebook.shape[0]
        self.codebook = torch.from_numpy(codebook).float().view(len_data, -1)
        if vary_label is not None:
            print("Not ReLU, vary to ", vary_label)
            self.codebook[self.codebook==0] = vary_label[0]
            self.codebook[self.codebook==1] = vary_label[1]
        self.coordinate = torch.from_numpy(coordinate).float()
        # 坐标归一化
        self.coordinate = (self.coordinate - self.coordinate.min(0)[0]) / \
                          (self.coordinate.max(0)[0] - self.coordinate.min(0)[0])  # 归一化数据到0-1
        if to11:
            self.coordinate = self.coordinate * 2 - 1 # 归一化数据到[1,-1]
        if PE is not None:
            print(PE)
            self.coordinate = PE(self.coordinate).view(len_data, -1)
        if to_cuda:
            self.coordinate=self.coordinate.to('cuda')
            self.codebook=self.codebook.to('cuda')
            print("check it to device")
    def __len__(self):
        return self.codebook.shape[0]

    def __getitem__(self, idx):
        return self.coordinate[idx], self.codebook[idx]


