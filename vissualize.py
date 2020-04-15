#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import argparse

import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import utils
from utils import PointLoss
from utils import distance_squre
import data_utils as d_utils
import ModelNet40Loader
import shapenet_part_loader
from Decoder import Decoder
from D_net import D_net
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10|ModelNet40|ShapeNet')
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=512,help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='/media/louise/ubuntu_work2/lost+found/dl_project/Trained_Model/gen_net0.pth', help="path to netG (to continue training)")
parser.add_argument('--infile',type = str, default = 'test_one/crop4-1.csv')
parser.add_argument('--infile_real',type = str, default = 'test_one/real4-1.csv')
parser.add_argument('--netD', default='/media/louise/ubuntu_work2/lost+found/dl_project/Trained_Model/dis_net0.pth', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
# If you want to test your point clouds.  Set the first parameter of '--point_scales_list' equal to (point_number + 512).
parser.add_argument('--point_scales_list',type=list,default=[2048,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
opt = parser.parse_args()
print(opt)

def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 


test_dset = shapenet_part_loader.PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice='Car', npoints=opt.pnum, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                         shuffle=True,num_workers = int(opt.workers))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gen_net = Decoder(opt.point_scales_list[0], opt.crop_point_num)
gen_net = torch.nn.DataParallel(gen_net)
gen_net.to(device)
gen_net.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])   
gen_net.eval()

input_cropped1 = np.loadtxt(opt.infile,delimiter=',')
input_cropped1 = torch.FloatTensor(input_cropped1)
input_cropped1 = torch.unsqueeze(input_cropped1, 0)
Zeros = torch.zeros(1,512,3)
input_cropped1 = torch.cat((input_cropped1,Zeros),1)
input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)

# input_cropped  = [input_cropped1,input_cropped2]
fake_center1,fake_fine = gen_net(input_cropped1)

#fake_center1,fake_center2, fake=gen_net(input_cropped1)
fake_fine = fake_fine.cuda()
fake_center1 = fake_center1.cuda()
#fake_center2 = fake_center2.cuda()


input_cropped2 = input_cropped2.cpu()
#input_cropped3 = input_cropped3.cpu()
#input_cropped4 = input_cropped4.cpu()

np_crop2 = input_cropped2[0].detach().numpy()
#np_crop3 = input_cropped3[0].detach().numpy()
#np_crop4 = input_cropped4[0].detach().numpy()

real = np.loadtxt(opt.infile_real,delimiter=',')
real = torch.FloatTensor(real)
real = torch.unsqueeze(real,0)
real2_idx = utils.farthest_point_sample(real,128, RAN = False)
real2 = utils.index_points(real,real2_idx)
#real3_idx = utils.farthest_point_sample(real,128, RAN = True)
#real3 = utils.index_points(real,real3_idx)

real2 = real2.cpu()
#real3 = real3.cpu()

np_real2 = real2[0].detach().numpy()
#np_real3 = real3[0].detach().numpy()


fake_fine = fake_fine.cpu()
fake_center1 = fake_center1.cpu()
#fake_center2 = fake_center2.cpu()
np_fake = fake_fine[0].detach().numpy()
np_fake1 = fake_center1[0].detach().numpy()
# np_fake2 = fake_center2[0].detach().numpy()
input_cropped1 = input_cropped1.cpu()
np_crop = input_cropped1[0].numpy() 

np.savetxt('test_one/crop_ours'+'.csv', np_crop, fmt = "%f,%f,%f")
np.savetxt('test_one/fake_ours'+'.csv', np_fake, fmt = "%f,%f,%f")
np.savetxt('test_one/crop_ours_txt'+'.txt', np_crop, fmt = "%f,%f,%f")
np.savetxt('test_one/fake_ours_txt'+'.txt', np_fake, fmt = "%f,%f,%f")
    
