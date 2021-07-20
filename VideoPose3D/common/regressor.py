
import torch

def regressor(camera, pose_pred, pose_2d, w, update):
    """create a linear regression in single view

    Args:
        camera (tensor): camera_parameters[cx,cy,fx,fy]
        pose_pred (tensor): predicted_3d_pose with shape[number,frame,joint,3]
        pose_2d (tensor): pixel_2d_pose with shape[number,frame,joint,3]
        w (int): number of frames in one shot
        update (bool): if compute loss
    """
    
    pose_pred = torch.split(pose_pred, split_size_or_sections=w, dim=1)
    pose_2d = torch.split(pose_2d, split_size_or_sections=w, dim=1)
    [subregressor(camera, i[0], i[1], update) for i in zip(pose_pred,pose_2d)]
    
    return

def subregressor(camera, pose_pred, pose_2d, update):
    """create a linear regression in single view with frame width w

    Args:
        camera (tensor): camera_parameters[cx,cy,fx,fy]
        pose_pred (tensor): predicted_3d_pose with shape[number,w,joint,3]
        pose_2d (tensor): pixel_2d_pose with shape[number,w,joint,3]
        update (bool): if compute loss
    """
    