import numpy as np
import argparse
import pandas as pd
import glob
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import utils
from mayavi import mlab

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10/ModelNet40/ShapeNet')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
opt = parser.parse_args()
print(opt)

# collate all car files into car_csv
#car_ids = [filename.split('.')[0] for filename in glob.glob('test_files/*.pcd')]
car_ids = ['test_files/car']
print(car_ids)
total_points = 0
for i, car_id in enumerate(car_ids):
    #test_npy = np.load(os.path.join('test_files/', car_id), allow_pickle = True)
    print(car_ids)
    #input_cropped1 = np.loadtxt(os.path.join('car_csv/', car_id.split('.')[0].split('/')[1]+'.csv'),delimiter=',')
    input_cropped1 = partial
    input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
    input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)

    fake_center1,fake_fine = gen_net(input_cropped1)

    fake_fine = fake_fine.cuda()
    fake_center1 = fake_center1.cuda()

    input_cropped2 = input_cropped2.cpu()

    real = torch.unsqueeze(real,0)
    real2_idx = utils.farthest_point_sample(real,128, RAN = False)
    real2 = utils.index_points(real,real2_idx)

    real2 = real2.cpu()

    np_real2 = real2[0].detach().numpy()

    vals='height'
    if vals == "height":
        col = z
    else:
        col = d
    
def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 

