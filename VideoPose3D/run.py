
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import os
import errno

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

try:
    # Create out directory if it does not exist
    os.makedirs(args.output)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create output directory:', args.checkpoint)


print('Loading dataset...')
dataset_path = 'data/data_multi_' + args.dataset + '.npz'
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
data_iter = DataLoader(dataset)
if args.output:
    output_zip = dict()

print('Processing...')

epoch = 0
pbar = tqdm(total=dataset.__len__())

while epoch < args.epochs:

    count = 0
    
    for cameras, pose_cs, pose_2ds in data_iter:

        if args.output:
            pose_pred = list()
            multi_T = list()
            output_zip[count] = dict()
            output_zip[count]['view_main'] = dict()

        if args.multi_view:
            raise KeyError('sorry, multi_view is not in beta test')

        # if have ground truth 3D pose, make a evaluation
        if not pose_cs and args.evaluate:
            raise KeyError('3D groung truth: 404 not found')
        
        if pose_cs and args.evaluate:
            pose_c_m = pose_cs[0].squeeze(0)

        # for main view
        pose_2d_m = pose_2ds[0].squeeze(0)
        camera_m = cameras[0].squeeze(0)

        # N - number of people
        N = pose_2d_m.shape[0]

        # for each person
        for i in range(N):
            pose_2d = pose_2d_m[i].unsqueeze(0)
            
            if args.update:
                model_pos.train()
                optimizer.zero_grad()
            else:
                model_pos.eval()

            pose_c_test = model_pos(pose_2d)
            pose_2d_test = pose_2d[:,receptive_field-1:]
            T, loss = regressor(pose_c_test, pose_2d_test, camera_m, args.update)

            if args.output:
                pose_pred.append(pose_c_test)
                multi_T.append(T)

            # if have ground truth 3D pose, make a evaluation
            if pose_cs and args.evaluate:
                # T -> [x,3] -> [1,x,17,3]
                T = T.unsqueeze(1)
                pose_c_gt = pose_c_m[i][receptive_field-1:]
                pose_c_test = pose_c_test.squeeze(0) + T
            
            if args.update:
                loss.backward()
                optimizer.step()
        
        if args.output:
            output_zip[count]['view_main']['pose_pred'] = pose_pred
            output_zip[count]['view_main']['T'] = multi_T
            output_zip[count]['view_main']['receptive_field'] = receptive_field
        
        count += 1
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
    output_filename = os.path.join(args.output, 'data/data_multi_output_' + args.dataset + '.npz')
    print('- Saving output to', output_filename)
    np.savez_compressed(output_filename, positions_3d=output_zip)
        
print('Done.')