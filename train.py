import os
import sys
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
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
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=12, help='input batch size')
parser.add_argument('--crop_point_num',type=int,default=512,help='0 means do not use else use with this weight')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=2,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[2048,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.95,help='0 means do not use else use with this weight')
parser.add_argument('--netG', default='', help="put in gen_net.pth location to continue training)")
parser.add_argument('--netD', default='', help="put in dis_net.pth location to continue training)")
opt = parser.parse_args()
# print(opt)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gen_net = Decoder(opt.point_scales_list[0], opt.crop_point_num)
dis_net = D_net(opt.crop_point_num)
cudnn.benchmark = True # faster runtime
resume_epoch = 0

print(dis_net)
print(gen_net)

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv2d") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("Conv1d") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)
	elif classname.find("BatchNorm1d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0) 

# initialize generator and discriminator weights and device
if USE_CUDA:
	print("Using", torch.cuda.device_count(), "GPUs")
	gen_net = torch.nn.DataParallel(gen_net)
	gen_net.to(device) 
	gen_net.apply(weights_init_normal)
	dis_net = torch.nn.DataParallel(dis_net)
	dis_net.to(device)
	dis_net.apply(weights_init_normal)

if opt.netG != '' :
	gen_net.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
	resume_epoch = torch.load(opt.netG)['epoch']
	#print('G loaded')
if opt.netD != '' :
	dis_net.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
	resume_epoch = torch.load(opt.netD)['epoch']
	#print('D loaded')
#print(resume_epoch)


# seeding for cropping
# manual seed and send seed to cuda
if opt.manualSeed is None:
	opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
	torch.cuda.manual_seed_all(opt.manualSeed)

# define transforms for point cloud data
transforms = transforms.Compose([d_utils.PointcloudToTensor(),])

train_set = shapenet_part_loader.PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.pnum, split='train')
assert train_set
train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize,
										 shuffle=True,num_workers = int(opt.workers))

test_set = shapenet_part_loader.PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.pnum, split='test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batchSize,
										 shuffle=True,num_workers = int(opt.workers))

criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion_PointLoss = PointLoss().to(device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = torch.optim.Adam(dis_net.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05,weight_decay=opt.weight_decay)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
optimizerG = torch.optim.Adam(gen_net.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05 ,weight_decay=opt.weight_decay)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

crop_point_num = int(opt.crop_point_num)
input_cropped1 = torch.FloatTensor(opt.batchSize, opt.pnum, 3)
label = torch.FloatTensor(opt.batchSize)

num_batch = len(train_set) / opt.batchSize

# train with generator and discriminator
for epoch in range(resume_epoch, opt.niter):
	if epoch<30:
		lam1 = 0.01
		lam2 = 0.02
	elif epoch<80:
		lam1 = 0.08
		lam2 = 0.1
	else:
		lam1 = 0.4
		lam2 = 0.5
	
	for i, data in enumerate(train_loader):
		real_point, target = data
		batch_size = real_point.size()[0] # real_point.shape = [24, 2024, 3]
		real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)       
		input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
		input_cropped1 = input_cropped1.data.copy_(real_point)
		real_point = torch.unsqueeze(real_point, 1)
		input_cropped1 = torch.unsqueeze(input_cropped1,1) # input_cropped1.shape = [24, 1, 2024, 3]
		p_origin = [0,0,0]
		label.resize_([batch_size,1]).fill_(real_label)
		if real_point.size()[0] < opt.batchSize: continue
		real_point = real_point.to(device) # real_point.shape = [24, 1, 2048, 3]
		real_center = real_center.to(device) # real_center.shape = [24, 1, 512, 3]
		input_cropped1 = input_cropped1.to(device) # input_cropped1.shape = [24, 1, 2048, 3]
		label = label.to(device) # real label construction done
		
		# obtain data for the two channels
		real_center = torch.squeeze(real_center,1) # [24, 512, 3]
		real_center_key1 = utils.index_points(real_center,real_center_key1_idx)

		input_cropped1 = torch.squeeze(input_cropped1,1)

		input_cropped = [input_cropped1, input_cropped2] # make sure if inputs are 2048 and 512
		gen_net = gen_net.train()
		dis_net = dis_net.train()

		# update discriminator
		dis_net.zero_grad()
		real_center = torch.unsqueeze(real_center,1)  
		print('real center shape', real_center.shape) 
		real_out = dis_net(real_center)
		#print('real label shape', label.shape)
		dis_err_real = criterion(real_out, label)
		dis_err_real.backward()

		fake_center1,fake_fine = gen_net(input_cropped1)
   
		errG_l2 = criterion_PointLoss(torch.squeeze(fake_fine,1),torch.squeeze(real_center,1))\
		+lam1*criterion_PointLoss(fake_center1,real_center_key1) # generator loss(AE loss)
		
		errG = opt.wtl2 * errG_D + opt.wtl2  # total adversarial loss
		#print('errG', errG)
		errG.backward()
		optimizerG.step()
		print('Epoch[%d/%d] Batch[%d/%d] D_loss: %.4f G_loss: %.4f errG_D: %.4f errG_l2: %.4f'
			  % (epoch, opt.niter, i, len(train_loader), 
				 dis_err.data, errG, errG_D.data, errG_l2))
		f=open('loss_PFNet.txt','a')
		f.write('\n'+'Epoch[%d/%d] Batch[%d/%d] D_loss: %.4f G_loss: %.4f errG_D: %.4f errG_l2: %.4f'
			  % (epoch, opt.niter, i, len(train_loader), 
				 dis_err.data, errG, errG_D.data, errG_l2))
	print('Training for epoch %d done'%(epoch))
	# start of testing
	if epoch % 5 == 0:
		print('After, ',epoch,'-th batch')
		for i, data in enumerate(test_loader):
			real_point, target = data
			batch_size = real_point.size()[0]
			if batch_size < opt.batchSize: continue
			real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
			input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
			input_cropped1 = input_cropped1.data.copy_(real_point)

			real_point = torch.unsqueeze(real_point, 1)
			input_cropped1 = torch.unsqueeze(input_cropped1,1)
				
			input_cropped1 = torch.squeeze(input_cropped1,1)

			input_cropped2 = input_cropped2.to(device)
			fake_center1, fake_fine = gen_net(input_cropped1)
			CD_loss = criterion_PointLoss(torch.squeeze(fake_fine,1),torch.squeeze(real_center,1))
			print('test CD loss: %.4f'%(CD_loss))
			f.write('\n'+'test result:  %.4f'%(CD_loss))
			break
		f.close()
		schedulerD.step()
		schedulerG.step()
		if epoch% 5 == 0:   
			torch.save({'epoch':epoch+1,
						'state_dict':gen_net.state_dict()},
						'Trained_Model/gen_net'+str(epoch)+'.pth')
			torch.save({'epoch':epoch+1,
						'state_dict':dis_net.state_dict()},
						'Trained_Model/dis_net'+str(epoch)+'.pth')

print('done')