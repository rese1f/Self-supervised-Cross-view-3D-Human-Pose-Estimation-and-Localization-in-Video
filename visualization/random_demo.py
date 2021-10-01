from star.star import STAR
import numpy as np
from numpy import newaxis
import pickle
import os
import torch


betas = np.array([
    np.array([2.25176191, -3.7883464, 0.46747496, 3.89178988,
              2.20098416, 0.26102114, -3.07428093, 0.55708514,
              -3.94442258, -2.88552087])]) # fixed
num_betas = 10
batch_size = 1
star = STAR(gender='female', num_betas=num_betas)

# Zero pose
# TODO: edit this part into poses
poses = torch.cuda.FloatTensor(np.zeros((batch_size, 72)))

betas = torch.cuda.FloatTensor(betas)
trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
model = star.forward(poses, betas, trans).cpu().detach().numpy()
# shaped = model.v_shaped[-1, :, :]

outmesh_path = 'objects/hello_star.obj'
with open(outmesh_path, 'w') as fp:
    for i in model:
        for v in i:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    for f in star.f + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
print('done.')