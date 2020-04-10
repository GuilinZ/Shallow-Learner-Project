import torch
import torch.nn as nn
import torch.nn.functional as F

# num_center_point: number of generated center points from dense output
class D_net(nn.Module):
	def __init__(self, num_center_point)
	self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
	self.conv2 = torch.nn.Conv2d(64, 64, 1)
    self.conv3 = torch.nn.Conv2d(64, 128, 1)
    self.conv4 = torch.nn.Conv2d(128, 256, 1)
    self.maxpool = torch.nn.MaxPool2d((self.num_center_point, 1), 1)

    self.bn1 = nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(64)
    self.bn3 = nn.BatchNorm2d(128)
    self.bn4 = nn.BatchNorm2d(256)

    self.fc1 = nn.Linear(448,256)
    self.fc2 = nn.Linear(256,128)
    self.fc3 = nn.Linear(128,16)

    self.fcbn1 = nn.BatchNorm1d(256)
    self.fcbn2 = nn.BatchNorm1d(128)
    self.fcbn3 = nn.BatchNorm1d(16)
    
    self.out = nn.Linear(16,1)

    def forward(self, x): # input size = [batch_size, num_center_point, 3]
    	x = F.relu(self.bn1(self.conv1(x)))
        x_low = F.relu(self.bn2(self.conv2(x)))
        x_low = torch.squeeze(self.maxpool(x_low))
        x_med = F.relu(self.bn3(self.conv3(x_low)))
        x_med = torch.squeeze(self.maxpool(x_med))
        x_high = F.relu(self.bn3(self.conv3(x_med)))
        x_high = torch.squeeze(self.maxpool(x_high))

        multi_layers = [x_high, x_med, x_low] 
        x = torch.cat(multi_layers, 1) # 256+128+64
        x = F.relu(self.fcbn1(self.fc1(x)))
        x = F.relu(self.fcbn2(self.fc2(x)))
        x = F.relu(self.fcbn3(self.fc3(x)))
        output = self.out(x)
        return output

# DISCRIMINATOR UPDATES TO BE WRITTEN TO MAIN
"""
criterion = torch.nn.BCEWithLogitsLoss().to(device)
optimizerD = torch.optim.Adam(D_net.parameters(), lr=0.0001, betas=(0.9, 0.999),eps=1e-05, weight_decay=0.001)

real_out = D_net(real_center) 						# assume real_center in shape [batch_size, num_center_point, 3]
errorD_real = criterion(real_out, real_label) 		# real_out and real_label in shape [batch_size, 1]
errorD_real.backward()

fake_out = D_net(fake_center)
errorD_fake = criterion(fake_out, fake_label)
errorD_fake.backward()
errorD = errorD_real + errorD_fake
optimizerD.step()
"""