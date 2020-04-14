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
    def __init__(self, num_points):
        super(Decoder, self).__init__()
        self.latentfeature = Encoder(num_points)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.fc1_1 = nn.Linear(1024, 128 * 512)
        self.fc2_1 = nn.Linear(512, 128 * 3)

        self.conv1_1 = torch.nn.Conv1d(512, 512, 1)
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, int((self.crop_point_num * 3) / 128), 1)

    def forward(self, x):
        x = self.latentfeature(x)
        x_1 = F.relu(self.fc1(x))  # 1024
        x_2 = F.relu(self.fc2(x_1))  # 512

        pc2_feat = self.fc2_1(x_2)
        pc2_xyz = pc2_feat.reshape(-1, 128, 3)  # 128x3 Global Points

        pc1_feat = F.relu(self.fc1_1(x_1))
        pc1_feat = pc1_feat.reshape(-1, 512, 128)
        pc1_feat = F.relu(self.conv1_1(pc1_feat))
        pc1_feat = F.relu(self.conv1_2(pc1_feat))
        pc1_xyz = self.conv1_3(pc1_feat)  # 12x128
        pc1_xyz = pc1_xyz.reshape(-1, 128, 4, 3)

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2) # 128x3x1
        pc2_xyz_expand = pc2_xyz_expand.transpose(1, 2) # 128x1x3
        pc1_xyz = pc1_xyz + pc2_xyz_expand # 128x4x3
        pc1_xyz = pc1_xyz.reshape(-1, 512, 3) # 512x3 Local Points

        return pc1_xyz, pc2_xyz