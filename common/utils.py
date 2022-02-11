# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import hashlib
import torch.nn.functional as F

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

def mask_generator(batch_size, output_len, num_joint):
    #mask = np.zeros((batch_size,num_joint,output_len))

    kernel_size = 13
    t_rand = torch.rand(batch_size, 1, output_len)
    filt = torch.ones([1, 1, kernel_size])/kernel_size

    t_rand = F.conv1d(t_rand, filt, padding=(kernel_size-1)//2)

    N_rand1 = torch.rand(batch_size, num_joint, 1)
    N_rand2 = torch.rand(batch_size, num_joint, 1)*0.15+0.3
    N_rand = N_rand1*(1-2*N_rand2)+N_rand2

    T_rand = torch.repeat_interleave(t_rand, num_joint, dim=1)
    N_rand = torch.repeat_interleave(N_rand, output_len, 2)
    #import pdb; pdb.set_trace()
    mask = (T_rand+N_rand)/2
    mask = torch.unsqueeze(mask, 1)

    mask = mask.detach().numpy()

    mask = np.transpose(mask, (0,3,2,1))

    tmp_rand2 = [0.55]# = np.random.uniform(0.4,0.6,1) # [0.5, 0.65]
    mask[mask>tmp_rand2[0]] = 1.
    mask[mask<=tmp_rand2[0]] = 0.
    mask[:,0,:,:] = 0.
    mask[:,-1,:,:] = 0.
    return mask