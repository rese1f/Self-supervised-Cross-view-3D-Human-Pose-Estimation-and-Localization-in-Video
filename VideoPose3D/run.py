
import torch
from torch.autograd.grad_mode import no_grad
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import os
import errno
import sys

from common.arguments import parse_args
from common.model import *
from common.regressor import *
from common.generators import ChunkedGenerator

args = parse_args()
print(args)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

if args.output:
    try:
        # Create out directory if it does not exist
        os.makedirs('output')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create output directory')


print('Loading dataset...')
dataset_path = 'data/data_multi_' + args.dataset + '.npz'
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
    print('- Running in device', torch.cuda.get_device_name())
    model_pos = model_pos.cuda()

receptive_field = model_pos.receptive_field()
lr = checkpoint['lr']
lr_decay = args.lr_decay  
initial_momentum = 0.1
final_momentum = 0.001
optimizer = optim.Adam(model_pos.parameters(), lr=lr, amsgrad=True)
if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])


print('Preparing data...')
dataset = ChunkedGenerator(dataset_zip)
data_iter = DataLoader(dataset, shuffle=True)

if args.output:
    output_zip = dict()
    loss_list = list()

print('Processing...')

epoch = 0
if args.update:
    model_pos.train()
else:
    model_pos.eval()
    
while epoch < args.epochs:
    print('- epoch {}'.format(epoch))
    pbar = tqdm(total=dataset.__len__())
    
    for cameras, pose_cs, pose_2ds, count in data_iter:
        # cut the useless dimention
        # pose - [view,number,frame,joint,2]
        # camera - [view,4]
        cameras = cameras.squeeze(0)
        pose_2ds = pose_2ds.squeeze(0)
        
        # if have ground truth 3D pose, make a evaluation
        if args.evaluate:
            if pose_cs == None:
                raise KeyError('3D groung truth: 404 not found')
            else:
                pose_cs = pose_cs.squeeze(0)
        
        # initial the output format
        if args.output and epoch==args.epochs-1:
            count = count.item()
            output_zip[count] = dict()

        if args.update:
            optimizer.zero_grad()   
        
        view_number = cameras.shape[0]
        # pose_pred - [view,number,frame,joint,3]
        pose_pred = torch.stack([model_pos(pose_2ds[v]) for v in range(view_number)])
        # for each view
        # here we make a cut for pose_2d via receptive_field
        T, loss = zip(*[regressor(cameras[v], pose_pred[v], pose_2ds[v,:,receptive_field-1:], args.width, args.update) for v in range(view_number)])
        
        # BETA
        T = torch.stack(T)
        if args.update:
            loss = torch.stack(loss).mean()
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        
        if args.output:
            output_zip['pose_pred'] = pose_pred
            output_zip['T'] = T
            output_zip['receptive_field'] = receptive_field
        
        pbar.update(1)
    pbar.close()

    # Decay learning rate exponentially
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
      
    # Decay BatchNorm momentum
    momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
    model_pos.set_bn_momentum(momentum)

    epoch += 1

    
if args.save:
    print('Saving model...')
    chk_path = os.path.join(args.checkpoint, args.save)
    print('- Saving checkpoint to', chk_path)
            
    torch.save({
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
    }, chk_path)

if args.output:
    print('Saving output...')
    output_filename = os.path.join('output/data_output_' + args.dataset + '_' + str(args.epochs) + '.npz')
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
            'receptive_field': int
        }
        sample'1': {
            'pose_pred': list(v,n,x,17,3),
            'T': list(v,n,x,3),
            'receptive_field': int
        }
    }
    # unit: m
    """

# Save training curves after every epoch, as .png images (if requested)
if args.export_training_curves:
    print('Saving traning curve...')
    if 'matplotlib' not in sys.modules:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    plt.figure()
    x = np.arange(0, len(loss_list))
    plt.plot(x, loss_list, linestyle='-', color='C0')
    plt.xlabel('Batch')
    plt.ylabel('Regression loss')
    plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))
    plt.close('all')

print('Done.')