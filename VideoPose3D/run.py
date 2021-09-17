
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from loguru import logger
import numpy as np
import os
import errno
import sys

from common.arguments import parse_args
from common.loss import *
from common.model import *
from common.regressor import *
from common.ground import *
from common.utils import *
from common.generators import ChunkedGenerator

torch.set_printoptions(precision=None, threshold=4096, edgeitems=None, linewidth=None, profile=None)

args = parse_args()
print(args)
logger.add('log/run.log')
logger.warning('iter_nums:{}'.format(args.iter_nums))

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

try:
    # Create out directory if it does not exist
    os.makedirs('log')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create output directory')


print('Loading dataset...')
dataset_path = '../data/data_multi_' + args.dataset + '.npz'
print('- Loading file', dataset_path)
dataset_zip = np.load(dataset_path, allow_pickle=True)['dataset']


print('Loading Model...')
filter_widths = [int(x) for x in args.architecture.split(',')]
model_pos = TemporalModel(args.keypoints_number, 2, args.keypoints_number, filter_widths, 
            args.causal, args.dropout, args.channels, args.dense)

chk_filename = os.path.join(args.checkpoint, args.load)
print('- Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
model_pos.load_state_dict(checkpoint['model_pos'])
if torch.cuda.is_available():
    print('- Running on device', torch.cuda.get_device_name())
    model_pos = model_pos.cuda()

receptive_field = model_pos.receptive_field()


print('Preparing data...')
dataset = ChunkedGenerator(dataset_zip)
data_iter = DataLoader(dataset, batch_size=1, shuffle=False)

output_zip = dict()

print('Processing...')

model_pos.eval()

# pbar = tqdm(total=dataset.__len__())
loss = list()

with torch.no_grad():
    for camera, pose, pose_2d, count in data_iter:
        # initial the output format      
        # cut the useless dimention
        # pose_2d - [view,number,frame,joint,2]
        # camera - [view,4] [cx,cy,fx,fy]
        # shape - [view,number,frame,joint,2]
        camera = camera.squeeze(0)
        pose = pose.squeeze(0)
        pose_2d = pose_2d.squeeze(0)
        shape = pose_2d.shape
        v, n, f, j, w = shape[0], shape[1], shape[2], shape[3], args.width
        # normolization
        # make a assignment x=(x-c)/w, y=(y-c)/w
        cameras = camera[:,None,None,None,:]
        pose_2d_temp = pose_2d.clone()
        pose_2d_temp[...,0].add_(-cameras[...,0]).mul_(1/cameras[...,0])
        pose_2d_temp[...,1].add_(-cameras[...,1]).mul_(-1/cameras[...,0])
        pose_2d_temp = pose_2d_temp.reshape(-1,f,j,2)
        pose_pred = model_pos(pose_2d_temp)
        # make scale trans
        scale = torch.mean(torch.div(0.25803840160369873,sk_len(pose_pred.unsqueeze(0))).squeeze(0),dim=-1).unsqueeze(-1).unsqueeze(-1)
        pose_pred *= scale
        del pose_2d_temp
        
        # regression  
        # make a assignment x=(x-c)/f, y=(y-c)/f
        pose_2d[...,0].add_(-cameras[...,0]).mul_(1/cameras[...,2])
        pose_2d[...,1].add_(-cameras[...,1]).mul_(1/cameras[...,3])
        # pose_2ds -> reshape to [view*number,frame,joint,2]
        pose_2d = pose_2d.reshape(-1,f,j,2)
        # here we make a cut for pose_2d via receptive_field
        pose_2d = pose_2d[:, (receptive_field-1)//2:-(receptive_field-1)//2]
        T, _ = init_regressor(pose_pred, pose_2d, w)
        T = T.reshape(v,n,-1,3)
        # compute ground equation
        foot = pose_pred.reshape(v,n,-1,j,3)[:,:,:,[3,6],:] + T.unsqueeze(3)
        # reshape to [v*f, n*2, 3]
        foot = foot.permute(0,2,1,3,4).reshape(-1,2*n,3)
        init_ground = ground_computer(foot)
        init_ground = init_ground.reshape(1,-1,3,1)
        T, ground, reg_loss = iter_regressor(pose_pred, pose_2d, init_ground, args.iter_nums, w)
        # reshape back to [view, number, frame, joint, 2/3]
        pose_2ds = pose_2d.reshape(v,n,-1,j,2)
        pose_pred = pose_pred.reshape(v,n,-1,j,3)
        pose_pred += T.unsqueeze(0).unsqueeze(3)
        pose = pose[:, :, (receptive_field-1)//2:-(receptive_field-1)//2]
        mpjpe_loss, scale = multi_n_mpjpe(pose_pred, pose)
        loss.append(mpjpe_loss)
        logger.info("id:{}, loss:{}, scale:{}".format(round(count.item(),4), round(mpjpe_loss.item(),4), round(scale[:,:,0].item(),4)))
        count = count.item()
        output_zip[count] = dict()
        output_zip[count]['pose_pred'] = pose_pred
        output_zip[count]['T'] = T
        output_zip[count]['ground'] = ground
        output_zip[count]['receptive_field'] = receptive_field
        output_zip[count]['scale'] = scale
        # logger.warning((pose[0,0,0,0,:]-T.unsqueeze(0).unsqueeze(3)[0,0,0,0,:]))
        # pbar.update(1)
        
# pbar.close()
logger.error('n_MPJPE:{}'.format(torch.mean(torch.stack(loss)).item()))

print('Saving output...')
output_filename = os.path.join('output/data_output_' + args.dataset + '_' + str(args.iter_nums) + '.npz')
print('- Saving output to', output_filename)
np.savez_compressed(output_filename, positions_2d=dataset_zip, positions_3d=output_zip)

"""
    dataset_zip = {
        sample'0': {

        }
    }
    # unit: mm

    output_zip = {
        sample'0': {
            'pose_pred': list(v,n,x,17,3),
            'T': list(v,n,x,3),
            'ground': [a,b,c],
            'receptive_field': int,
            'scale': list(v,n,x)
        }
        sample'1': {
            'pose_pred': list(v,n,x,17,3),
            'T': list(v,n,x,3),
            'ground': [a,b,c],
            'receptive_field': int,
            'scale': list(v,n,x)
        }
    }
    # unit: m
"""