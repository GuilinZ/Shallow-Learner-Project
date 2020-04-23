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
from utils import PointLoss
from utils import distance_squre
import data_utils as d_utils
import ModelNet40Loader
import shapenet_part_loader
from Decoder import Decoder
from D_net import D_net
from io_util import read_pcd, save_pcd
from mayavi import mlab

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10/ModelNet40/ShapeNet')
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
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='/media/louise/ubuntu_work2/lost+found/dl_project/Trained_Model/gen_net145.pth', help="path to gen_net")
parser.add_argument('--infile',type = str, default = 'car_csv/frame_425_car_0.csv')
parser.add_argument('--infile_real',type = str, default = 'test_files/real4-1.csv')
parser.add_argument('--netD', default='/media/louise/ubuntu_work2/lost+found/dl_project/Trained_Model/dis_net145.pth', help="path to dis_net")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[1921,512],help='number of points in each scales') # set the first number of '--point_scales_list' to (point_number + 512).
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
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
    partial = read_pcd('%s.pcd' % car_id)
    """
    bbox = np.loadtxt( '%s.txt' % car_id)
    total_points += partial.shape[0]

    center = (bbox.min(0) + bbox.max(0)) / 2
    bbox -= center
    yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
    bbox = np.dot(bbox, rotation)
    scale = bbox[3, 0] - bbox[0, 0]
    bbox /= scale

    partial = np.dot(partial - center, rotation) / scale
    partial = np.dot(partial, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    np.savetxt(os.path.join('car_csv/', car_id.split('.')[0].split('/')[1]+'.csv'), partial, delimiter=",")
    print('total_points', total_points)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen_net = Decoder(opt.point_scales_list[0], opt.crop_point_num)
    gen_net = torch.nn.DataParallel(gen_net)
    gen_net.to(device)
    gen_net.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])   
    gen_net.eval()

    #input_cropped1 = np.loadtxt(os.path.join('car_csv/', car_id.split('.')[0].split('/')[1]+'.csv'),delimiter=',')
    input_cropped1 = partial
    input_cropped1 = torch.FloatTensor(input_cropped1)
    input_cropped1 = torch.unsqueeze(input_cropped1, 0)
    print(input_cropped1.shape)
    Zeros = torch.zeros(1,512,3)
    input_cropped1 = torch.cat((input_cropped1,Zeros),1)
    print(input_cropped1.shape)
    input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
    input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)

    fake_center1,fake_fine = gen_net(input_cropped1)

    fake_fine = fake_fine.cuda()
    fake_center1 = fake_center1.cuda()

    input_cropped2 = input_cropped2.cpu()

    np_crop2 = input_cropped2[0].detach().numpy()

    real = np.loadtxt(opt.infile_real,delimiter=',')
    real = torch.FloatTensor(real)
    real = torch.unsqueeze(real,0)
    real2_idx = utils.farthest_point_sample(real,128, RAN = False)
    real2 = utils.index_points(real,real2_idx)

    real2 = real2.cpu()

    np_real2 = real2[0].detach().numpy()

    fake_fine = fake_fine.cpu()
    fake_center1 = fake_center1.cpu()
    np_fake = fake_fine[0].detach().numpy()
    np_fake1 = fake_center1[0].detach().numpy()
    input_cropped1 = input_cropped1.cpu()
    np_crop = input_cropped1[0].numpy() 
    print(np_fake.shape, np_fake1.shape, np_crop.shape)
    display = np.vstack((np_fake, np_fake1, np_crop))
    x = display[:, 0]  # x position of point
    y = display[:, 1]  # y position of point
    z = display[:, 2]  # z position of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
     
    vals='height'
    if vals == "height":
        col = z
    else:
        col = d
     
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mlab.points3d(x, y, z,
                         col,          # Values used for Color
                         mode="point",
                         colormap='spectral', # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )
    mlab.show()
    """
    df = pd.DataFrame({'X': test_npy[:,0],
                       'Y': test_npy[:,1],
                       'Z': test_npy[:,2]})
    df['X'] = df.X #- df.X.min()
    df['Y'] = df.Y #- df.Y.min()
    df['Z'] = df.Z #- df.Z.min()
    df['XN'] = df.X#(df.X - df.X.mean()) / (df.X.max() - df.X.min())
    df['YN'] = (df.Y - df.Y.mean()) / (df.Y.max() - df.Y.min())
    df['ZN'] = (df.Z - df.Z.mean()) / (df.Z.max() - df.Z.min())
    header = ["XN", "YN", "ZN"]
    df[['XN','YN','ZN']].to_csv(os.path.join('car_csv/', car_id.split('.')[0]+'.csv'), header = False, index = False)
    """
def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 

