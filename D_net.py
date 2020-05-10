import torch
import torch.nn as nn
import torch.nn.functional as F

# num_center_point: number of generated center points from dense output
class D_net(nn.Module):
	def __init__(self, num_center_point):
		super(D_net, self).__init__()
		self.num_center_point = num_center_point
		self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
		self.conv2 = torch.nn.Conv2d(64, 64, 1)
		self.conv3 = torch.nn.Conv2d(64, 128, 1)
		self.maxpool = torch.nn.MaxPool2d((512, 1), 1)
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(128)
		self.fc1 = nn.Linear(448,256)
		self.fc2 = nn.Linear(256,64)
		self.fc3 = nn.Linear(64,16)
		self.bn_1 = nn.BatchNorm1d(256)
		self.bn_2 = nn.BatchNorm1d(128)
		self.bn_3 = nn.BatchNorm1d(16)

	def forward(self, x): # input size = [batch_size, num_center_point, 3]
		#print('discriminator input', x.shape)
		x = F.relu(self.bn1(self.conv1(x)))
		x_64 = F.relu(self.conv2(x))
		x_128 = F.relu(self.conv3(x_64))
		x_64 = torch.squeeze(x_64)
		x_128 = torch.squeeze(x_128)
		x_256 = torch.squeeze(x_256)
		Layers = [x_256,x_128,x_64]
		x = torch.cat(Layers,1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		#print('discriminator output', x.shape)
		return x