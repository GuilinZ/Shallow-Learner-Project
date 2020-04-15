import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch
from pointnet_util import PointNetSetAbstraction

class Encoder(nn.Module):
	def __init__(self, num_points):
		super(Encoder, self).__init__()
		self.fe1 = FeatureExtractor_1()
		self.fe2 = FeatureExtractor_2(512)
		self.out_layer = nn.MaxPool2d((1, 2), 1)

	def forward(self, x):
		#print(x)
		x_down, out_1 = self.fe1(x) # (batch_size, 512, 3) || (batch_size, 1920)
		out_2 = self.fe2(x_down) # (batch_size, 1920)
		out = torch.cat((out_1, out_2), 2) # (batch_size, 1920, 2)
		#print('cat', out.shape)
		out = self.out_layer(out).view(-1, 1920) # (batch_size, 1920)
		#print('encoder output shape', out.shape)
		return out

	def downsampling(self, x): # (batch_size, 2048, 3)
		pass

class FeatureExtractor_1(nn.Module):
	def __init__(self):
		super(FeatureExtractor_1, self).__init__()
		self.in_channel = 3
		self.sa1 = PointNetSetAbstraction(npoint = 1024, radius = 0.2, nsample = 32,
										  in_channel = self.in_channel, mlp = [64, 64, 128], group_all = False)
		self.sa2 = PointNetSetAbstraction(npoint = 512, radius = 0.2, nsample = 32, in_channel = 128 + self.in_channel,
										  mlp = [128, 128, 256], group_all = False)
		self.sa3 = PointNetSetAbstraction(npoint = 128, radius = 0.2, nsample = 32, in_channel = 256 + self.in_channel,
										  mlp = [256, 256, 512], group_all = False)
		self.sa4 = PointNetSetAbstraction(npoint = 1, radius = 0.2, nsample = 32, in_channel = 512 + self.in_channel,
										  mlp = [512, 512, 1920], group_all = False)

	def forward(self, xyz):
		xyz = xyz.permute(0, 2, 1)
		batch_size, _, _ = xyz.shape
		norm = None
		l1_xyz, l1_points = self.sa1(xyz, norm)
		#print('xyz after sa1', l1_xyz.shape, l1_points.shape)
		l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

		#print('xyz after sa2', l2_xyz.shape, l2_points.shape)
		l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

		#print('xyz after sa3', l3_xyz.shape, l3_points.shape)
		l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

		#print('xyz after sa4', l4_xyz.shape, l4_points.shape)

		#output = l4_points.view(batch_size, 1920)
		output = l4_points
		return l2_xyz, output # (batch_size, 1920)

# num_center_point: number of generated center points from dense output
class FeatureExtractor_2(nn.Module):
	def __init__(self, num_points):
		super(FeatureExtractor_2, self).__init__()
		self.num_points = num_points
		self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
		self.conv2 = torch.nn.Conv2d(64, 64, 1)
		self.conv3 = torch.nn.Conv2d(64, 128, 1)
		self.conv4 = torch.nn.Conv2d(128, 256, 1)
		self.conv5 = torch.nn.Conv2d(256, 512, 1)
		self.conv6 = nn.Conv2d(512, 1024, 1)
		self.maxpool = nn.MaxPool2d((self.num_points, 1), 1)

		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(128)
		self.bn4 = nn.BatchNorm2d(256)
		self.bn5 = nn.BatchNorm2d(512)
		self.bn6 = nn.BatchNorm2d(1024)

		self.fc1 = nn.Linear(1920, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 128)

		self.fcbn1 = nn.BatchNorm1d(512)
		self.fcbn2 = nn.BatchNorm1d(256)
		self.fcbn3 = nn.BatchNorm1d(128)


	def forward(self, x): # input size = [batch_size, 3, num_center_point]
		#print('input size', x.shape)
		x = x.permute(0, 2, 1) # [batch_size, num_center_point, 3]
		x = torch.unsqueeze(x, 1)
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x_128 = F.relu(self.bn3(self.conv3(x)))
		#print(x_128.shape)
		x_256 = F.relu(self.bn4(self.conv4(x_128)))
		#print(x_256.shape)
		x_512 = F.relu(self.bn5(self.conv5(x_256)))
		#print(x_512.shape)
		x_1024 = F.relu(self.bn6(self.conv6(x_512))) 
		#print(x_1024.shape)

		x_128 = torch.squeeze(self.maxpool(x_128), 2) 
		x_256 = torch.squeeze(self.maxpool(x_256), 2)
		x_512 = torch.squeeze(self.maxpool(x_512), 2)
		x_1024 = torch.squeeze(self.maxpool(x_1024), 2)
		multi_layers = [x_1024, x_512, x_256, x_128]

		output = torch.cat(multi_layers, 1) # 1024+512+256+128 = 1920
		#print(output.shape)
		return output # (batch_size, 1920, 1)

