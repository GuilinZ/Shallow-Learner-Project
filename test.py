# Shallow-Learner-Project

#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import argparse
import random
from torch.autograd import Variable

import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import utils
from utils import PointLoss_test
from utils import distance_squre
import data_utils as d_utils
import ModelNet40Loader
import shapenet_part_loader
from Decoder import Decoder
from D_net import D_net
from mayavi import mlab
from io_util import read_pcd, save_pcd

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10|ModelNet40|ShapeNet')
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=1024,help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='/media/louise/ubuntu_work2/lost+found/dl_project/Trained_Model/1024_cut_num/gen_net170.pth', help="path to netG (to continue training)")
parser.add_argument('--infile',type = str, default = 'test_files/crop12.csv')
parser.add_argument('--infile_real',type = str, default = 'test_files/real11.csv')
parser.add_argument('--netD', default='/media/louise/ubuntu_work2/lost+found/dl_project/Trained_Model/dis_net0.pth', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
# Set the first parameter of '--point_scales_list' equal to (point_number + 512).
parser.add_argument('--point_scales_list',type=list,default=[2048,1024],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
opt = parser.parse_args()
print(opt)

def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dset = shapenet_part_loader.PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice='Airplane', npoints=opt.pnum, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                         shuffle=True,num_workers = int(opt.workers))
criterion_PointLoss = PointLoss_test().to(device)
losses1, losses2 = [], []
print(len(test_dataloader))
for i, data in enumerate(test_dataloader):
	real_point, target = data
	batch_size = real_point.size()[0]
	if batch_size < opt.batchSize: continue
	real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
	input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
	input_cropped1 = input_cropped1.data.copy_(real_point)

	real_point = torch.unsqueeze(real_point, 1)
	input_cropped1 = torch.unsqueeze(input_cropped1,1)

	p_origin = [0,0,0]

	if opt.cropmethod == 'random_center':
		choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
		
		for m in range(batch_size):
			index = random.sample(choice,1)
			distance_list = []
			p_center = index[0]
			for n in range(opt.pnum):
				distance_list.append(distance_squre(real_point[m,0,n],p_center))
			distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])                         
			for sp in range(opt.crop_point_num):
				input_cropped1.data[m,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])
				real_center.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]  
	real_center = real_center.to(device)
	real_center = torch.squeeze(real_center,1)
	input_cropped1 = input_cropped1.to(device) 
	input_cropped1 = torch.squeeze(input_cropped1,1)
	input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
	input_cropped2 = utils.index_points(input_cropped1,input_cropped2_idx)

	input_cropped1 = Variable(input_cropped1,requires_grad = False)
	input_cropped2 = Variable(input_cropped2,requires_grad = False)
	input_cropped2 = input_cropped2.to(device)

	gen_net = Decoder(opt.point_scales_list[0], opt.crop_point_num)
	gen_net = torch.nn.DataParallel(gen_net)
	gen_net.to(device)
	gen_net.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])   
	gen_net.eval()

	fake_center1, fake_fine = gen_net(input_cropped1)
	CD_loss_all, dist1, dist2 = criterion_PointLoss(torch.squeeze(fake_fine,1),torch.squeeze(real_center,1))
	#print('test CD loss: %.4f'%(dist1.item()))
	print('pred->GT|GT->pred:', dist1.item(), dist2.item())
	losses1.append(dist1.item())
	losses2.append(dist2.item())

print('mean CD loss pred->GT|GT->pred:', np.mean(losses1)*1000, np.mean(losses2)*1000)
print('max CD loss pred->GT|GT->pred: ', np.amax(losses1)*1000, np.amax(losses2)*1000)
print('min CD loss: pred->GT|GT->pred', np.amin(losses1)*1000, np.amin(losses2)*1000)


