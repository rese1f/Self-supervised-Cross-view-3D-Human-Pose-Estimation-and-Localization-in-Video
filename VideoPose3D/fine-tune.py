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
while epoch < 5:
    loss_list = []
    for camera, pose, pose_2d, count in data_iter:
        camera, pose, pose_2d = camera.squeeze(0), pose.squeeze(0), pose_2d.squeeze(0)
        shape = pose_2d.shape
        v, n, f, j = shape[0], shape[1], shape[2], shape[3]
        cameras = camera[:,None,None,None,:]
        pose_2dt = pose_2d.clone()
        pose_2dt[...,0].add_(-cameras[...,0]).mul_(1/cameras[...,0])
        pose_2dt[...,1].add_(-cameras[...,1]).mul_(-1/cameras[...,0])
        pose_2dt = pose_2dt.reshape(-1, f, j, 2)
        pose_pred = model_pos(pose_2dt)
        scale = torch.mean(torch.div(0.25803840160369873,sk_len(pose_pred.unsqueeze(0))).squeeze(0),dim=-1).unsqueeze(-1).unsqueeze(-1)
        pose_pred *= scale
        pose_pred = pose_pred.reshape(v, n, -1, j, 3)
        # pose -= pose[:,:,:,0].unsqueeze(3)
        # pose = pose[:,:,121:-121]
        pose_2d = pose_2d[:,:,121:-121]
        pose_2d[...,0].add_(-cameras[...,0]).mul_(1/cameras[...,2])
        pose_2d[...,1].add_(-cameras[...,1]).mul_(1/cameras[...,3])
        k1, k2 = 1, 1e5
        loss1, w = bone_loss(pose_pred)
        loss2 = projection_loss(pose_pred, pose_2d, camera)
        loss = k1*loss1 + k2*loss2
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        loss = loss.item()
        logger.info("loss: {}, loss1: {}, loss2: {}".format(loss, loss1, loss2))
        loss_list.append(loss)
    logger.error("epoch: {}, loss: {}".format(epoch, np.mean(loss_list)))
    epoch += 1
    
chk_path = './checkpoint/ft.bin'
torch.save({
    'epoch': epoch,
    'optimizer': optimizer.state_dict(),
    'model_pos': model_pos.state_dict(),
    }, chk_path)