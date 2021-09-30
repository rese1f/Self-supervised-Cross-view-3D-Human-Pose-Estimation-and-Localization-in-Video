# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
from common.utils import *
from common.regressor import *

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def multi_n_mpjpe(predicted, target):
    """
    [view(1),id(4-5),f,17,3]
    Multi person and view Normalized MPJPE (scale only)
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=4, keepdim=True), dim=3, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=4, keepdim=True), dim=3, keepdim=True)
    scale = torch.mean(norm_target, dim=1, keepdim=True)/torch.mean(norm_predicted, dim=1, keepdim=True)
    return mpjpe(scale * predicted, target), scale

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))

def bone_loss(pose):
    """loss due to length ratio of bone

    Args:
        pose (v,n,f,17,3)
        
    Returns:
        loss
    """
    H = torch.mean(sk_len(pose).reshape(-1,16), dim=0).unsqueeze(-1)
    y = torch.tensor([0.1318, 0.4513, 0.4455, 0.1318, 0.4508, 0.4460, 0.2412, 0.2553, 0.1162, 0.1147, 0.1472, 0.2806, 0.2448, 0.1479, 0.2774, 0.2463]).cuda().unsqueeze(-1)
    w = torch.mm(torch.mm(torch.mm(H.T,H).inverse(),H.T),y)
    loss = torch.norm(H*w-y)
    return loss, w

def projection_loss(pose_pred, pose_2d, camera):
    """loss due to 2D-3D projection

    Args:
        pose_pred (v,n,f,17,3)
        pose_2d (v,n,f,17,2)
        
    Returns:
        loss
    """
    cx, cy, fx, fy = camera[:,0], camera[:,1], camera[:,2], camera[:,3]
    f = pose_2d.shape[2]
    pose_pred = pose_pred.reshape(-1,f,17,3)
    pose_2d = pose_2d.reshape(-1,f,17,2)
    T, loss = init_regressor(pose_pred, pose_2d, 128)
    return loss