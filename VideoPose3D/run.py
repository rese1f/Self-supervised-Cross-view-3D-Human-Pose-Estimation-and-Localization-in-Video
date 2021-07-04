import torch
from auto_dataset import MultiDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch.optim as optim
import numpy as np
import sys
import errno

from arguments import parse_args
from auto_dataset import *
from model import *
from loss import *


args = parse_args()
print(args)


try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading model...')
filter_widths = [int(x) for x in args.architecture.split(',')]
model_pos_train = TemporalModel(args.joints_number, 2, args.joints_number,
                            filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)
model_pos = TemporalModel(args.joints_number, 2, args.joints_number,
                            filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)

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

lr = args.learning_rate
optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)
lr_decay = args.lr_decay

losses_3d_train = []

initial_momentum = 0.1
final_momentum = 0.001

print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
print('** The final evaluation will be carried out after the last training epoch.')

path = r'../DataGenerators/output'
dataset = MultiDataset(path)
train_dataset = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

for epoch in tqdm(range(args.epochs)):
    epoch_loss_3d_train = 0
    N = 0
    model_pos_train.train()

    for inputs_3d, inputs_2d in train_dataset:
        inputs_3d = torch.from_numpy(inputs_3d)
        inputs_2d = torch.from_numpy(inputs_2d)
        if torch.cuda.is_available():
            inputs_3d = inputs_3d.cuda()
            inputs_2d = inputs_2d.cuda()
        optimizer.zero_grad()
        # Predict 3D poses
        predicted_3d_pos = model_pos_train(inputs_2d)
        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
        epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
        N += inputs_3d.shape[0]*inputs_3d.shape[1]

        loss_total = loss_3d_pos
        loss_total.backward()

        optimizer.step()

    losses_3d_train.append(epoch_loss_3d_train / N)

    print('[%d] time %.2f lr %f 3d_train %f' % (
            epoch + 1,
            lr,
            losses_3d_train[-1]))

    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay

    momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
    model_pos_train.set_bn_momentum(momentum)

    # Save training curves after every epoch, as .png images (if requested)
    if args.export_training_curves and epoch > 3:
        if 'matplotlib' not in sys.modules:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
        plt.figure()
        epoch_x = np.arange(3, len(losses_3d_train)) + 1
        plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
        plt.legend(['3d train'])
        plt.ylabel('MPJPE (m)')
        plt.xlabel('Epoch')
        plt.xlim((3, epoch))
        plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))