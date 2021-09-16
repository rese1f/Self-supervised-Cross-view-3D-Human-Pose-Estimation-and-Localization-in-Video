import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
from loguru import logger

from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator

torch.set_printoptions(precision=None, threshold=4096, edgeitems=None, linewidth=None, profile=None)
logger.add('log/fine-tune.log')

pose_checkpoint = torch.load('checkpoint/pretrained_h36m_cpn.bin', map_location=lambda storage, loc: storage)
model_pos = TemporalModel(17, 2, 17, filter_widths=[3,3,3,3,3], causal=False, dropout=0.25, channels=1024, dense=False)
model_pos.load_state_dict(pose_checkpoint['model_pos'])
if torch.cuda.is_available():
    model_pos = model_pos.cuda()
model_pos.train()
dataset_path = '../data/data_multi_h36m.npz'
dataset_zip = np.load(dataset_path, allow_pickle=True)['dataset']
dataset = ChunkedGenerator(dataset_zip)
data_iter = DataLoader(dataset, batch_size=1, shuffle=False)
optimizer = optim.Adam(list(model_pos.parameters()) ,lr=1e-3, amsgrad=True)
if 'optimizer' in pose_checkpoint and pose_checkpoint['optimizer'] is not None:
    optimizer.load_state_dict(pose_checkpoint['optimizer'])

epoch = 0
while epoch < 50:
    loss_list = []
    for camera, pose, pose_2d, count in data_iter:
        camera, pose, pose_2d = camera.squeeze(0), pose.squeeze(0), pose_2d.squeeze(0)
        pose = pose[:,:,121:-121]
        shape = pose_2d.shape
        v, n, f, j = shape[0], shape[1], shape[2], shape[3]
        cameras = camera[:,None,None,None,:]
        pose_2d[...,0].add_(-cameras[...,0]).mul_(1/cameras[...,0])
        pose_2d[...,1].add_(-cameras[...,1]).mul_(-1/cameras[...,0])
        pose_2d = pose_2d.reshape(-1,f,j,2)
        pose_pred = model_pos(pose_2d).reshape(1, -1, j, 3)
        pose -= pose[:,:,:,0].unsqueeze(3)
        pose = pose.reshape(1, -1, j, 3)
        loss = n_mpjpe(pose_pred, pose)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        logger.info("loss: {}".format(loss))
        loss_list.append(loss)
    logger.error("epoch: {}, loss: {}".format(epoch, np.mean(loss_list)))
    loss_list = []
    epoch += 1
    
chk_path = './checkpoint/ft.bin'
torch.save({
    'epoch': epoch,
    'optimizer': optimizer.state_dict(),
    'model_pos': model_pos.state_dict(),
    }, chk_path)