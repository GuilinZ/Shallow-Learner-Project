import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch
from pointnet_util import PointNetSetAbstraction
from Encoder import Encoder

class Decoder(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list, crop_point_num):
        super(Decoder, self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Encoder(num_scales, each_scales_size, point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.fc1_1 = nn.Linear(1024, 128 * 512)
        self.fc2_1 = nn.Linear(512, 64 * 128)  # nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256, 64 * 3)

        #        self.bn1 = nn.BatchNorm1d(1024)
        #        self.bn2 = nn.BatchNorm1d(512)
        #        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
        #        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
        #        self.bn5 = nn.BatchNorm1d(64*128)
        #
        self.conv1_1 = torch.nn.Conv1d(512, 512, 1)  # torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, int((self.crop_point_num * 3) / 128), 1)
        self.conv2_1 = torch.nn.Conv1d(128, 6, 1)

    def forward(self, x):
        x = self.latentfeature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)