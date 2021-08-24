  
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>
#
#
# Code Developed by:
# Ahmed A. A. Osman
from star.star import STAR
import numpy as np
from numpy import newaxis
import pickle
import os
import torch

betas = np.array([
            np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
                      2.20098416, 0.26102114, -3.07428093, 0.55708514,
                      -3.94442258, -2.88552087])])
num_betas=10
batch_size=1
star = STAR(gender='female',num_betas=num_betas)

# Zero pose
poses = torch.cuda.FloatTensor(np.zeros((batch_size,72)))
betas = torch.cuda.FloatTensor(betas)

trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
model = star.forward(poses, betas,trans).cpu().detach().numpy()
# shaped = model.v_shaped[-1, :, :]

outmesh_path = './hello_star.obj'
with open(outmesh_path, 'w') as fp:
    for i in model:
        for v in i:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
    for f in star.f+1:
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
print('done.')