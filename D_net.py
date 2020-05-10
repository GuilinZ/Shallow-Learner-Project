import torch
import torch.nn as nn
import torch.nn.functional as F

# num_center_point: number of generated center points from dense output
class D_net(nn.Module):
	def __init__(self, num_center_point):
		super(D_net, self).__init__()
		self.num_center_point = num_center_point
		self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
		self.conv2 = torch.nn.Conv2d(64, 128, 1)
		self.fc3 = nn.Linear(64,16)
		self.bn_1 = nn.BatchNorm1d(256)
		self.bn_2 = nn.BatchNorm1d(128)
	def forward(self, x): # input size = [batch_size, num_center_point, 3]
		# print('discriminator input', x.shape)
		batch_size = x.size()[0]
		x = F.relu(self.bn1(self.conv1(x)))
		x_64 = F.relu(self.conv2(x))
		x_128 = F.relu(self.conv3(x_64))
		x_64 = torch.squeeze(x_64)
		x_128 = torch.squeeze(x_128)
		x_256 = torch.squeeze(x_256)
		# size asserts
		print(x_64.shape, x_128.shape, x_256.shape)
		if len(x_64.shape)==1: x_64 = x_64.view(batch_size, -1)
		if len(x_128.shape)==1: x_128 = x_128.view(batch_size, -1)
		if len(x_256.shape)==1: x_256 = x_256.view(batch_size,-1)

		Layers = [x_256, x_128, x_64]
		x = torch.cat(Layers, 1)
		x = F.relu(self.bn_1(self.fc1(x)))
		x = F.relu(self.bn_2(self.fc2(x)))
		x = self.fc4(x)
		#print('discriminator output', x.shape)
		return x