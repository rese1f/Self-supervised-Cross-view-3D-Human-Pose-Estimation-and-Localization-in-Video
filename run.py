import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random
import matplotlib.pyplot as plt

def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
                
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                
            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])
                
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
    
    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    
    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    
    return out_camera_params, out_poses_3d, out_poses_2d

if __name__ == '__main__':
    args = parse_args()
    print(args)

    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

    print('Loading dataset...')
    dataset_path = 'data/data_3d_' + args.dataset + '.npz'
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
            
    print('Preparing data...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    print('Loading 2D detections...')
    keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue
                
            for cam_idx in range(len(keypoints[subject][action])):
                
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])
            
    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    subjects_train = args.subjects_train.split(',')
    subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
    if not args.render:
        subjects_test = args.subjects_test.split(',')
    else:
        subjects_test = [args.viz_subject]
        
    
    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)
        
    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

    filter_widths = [int(x) for x in args.architecture.split(',')]
    if not args.disable_optimizations and not args.dense and args.stride == 1:
        # Use optimized model for single-frame predictions
        model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)
    else:
        # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
        model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                    dense=args.dense)
        
    model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                dense=args.dense)

    init_model = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                dense=args.dense)
    init_model = init_model.cuda()
    #import pdb; pdb.set_trace()
    receptive_field = model_pos.receptive_field()
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field - 1) // 2 # Padding on each side
    if args.causal:
        print('INFO: Using causal convolutions')
        causal_shift = pad
    else:
        causal_shift = 0

    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
        model_pos_train = model_pos_train.cuda()
        
    if args.resume or args.evaluate:
        chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
        print('Loading checkpoint', chk_filename)
        #import pdb; pdb.set_trace()
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        init_ckpt = torch.load('checkpoint/pretrained_h36m_cpn.bin', map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        model_pos_train.load_state_dict(checkpoint['model_pos'])
        model_pos.load_state_dict(checkpoint['model_pos'])
        init_model.load_state_dict(init_ckpt['model_pos'])
        #import pdb; pdb.set_trace()
        
    test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)
    lr = args.learning_rate
    optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)
    lr_decay = args.lr_decay

    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001
    
    
    train_generator = ChunkedGenerator(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))
    
    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        
        lr = checkpoint['lr']
            
    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')
    
    # Pos model only
    epoch = 0
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        init_model.eval()
        model_pos_train.eval()
        
        for batch_cam, batch_3d, batch_2d in train_generator.next_epoch():
            cam = torch.from_numpy(batch_cam.astype('float32'))
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            
            if torch.cuda.is_available():
                cam = cam.cuda()
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                
            inputs_traj = inputs_3d[:, :, :1].clone()
            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos = init_model(inputs_2d)
            predicted_3d_traj = model_pos_train(inputs_2d)
            
            dlt = DLT(cam)
            traj, R = dlt(predicted_3d_pos, inputs_2d)
            import pdb
            pdb.set_trace()
            w = 1 / inputs_traj[:, :, :, 2] # Weight inversely proportional to depth
            loss_3d_pos = weighted_mpjpe(predicted_3d_pos, inputs_traj, w)                
            #loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            #import pdb; pdb.set_trace()

            epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0]*inputs_3d.shape[1]

            loss_total = loss_3d_pos
            # loss_total.backward()

            optimizer.step()

            #break
            
        elapsed = (time() - start_time)/60
        
        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000))
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1
        
        # Decay BatchNorm momentum
        momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        model_pos_train.set_bn_momentum(momentum)
            
        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'a_epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)
            
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
            }, chk_path)