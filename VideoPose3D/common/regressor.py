
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
    N = pose_2d.shape[0]
    # make a assignment x=(x-c)/f, y=(y-c)/f
    pose_2d[..., 0] = (pose_2d[..., 0] - camera[0]) / (camera[2]*1e-3)
    pose_2d[..., 1] = (pose_2d[..., 1] - camera[1]) / (camera[3]*1e-3)

    # split via person and width
    pose_pred = torch.split(pose_pred, split_size_or_sections=w, dim=1)
    pose_2d = torch.split(pose_2d, split_size_or_sections=w, dim=1)
    [subregressor(i[0][j], i[1][j], update) for j in range(N) for i in zip(pose_pred,pose_2d)]
    
    return

def subregressor(pose_pred, pose_2d, update):
    """create a linear regression in single person with frame width w

    Args:
        pose_pred (tensor): predicted_3d_pose with shape[w,joint,3]
        pose_2d (tensor): pixel_2d_pose with shape[w,joint,3]
        update (bool): if compute loss
    """
    # initial the matrix
    w = pose_2d.shape[0]
    A = torch.zeros((40*w-9,3*w))
    b = torch.zeros((40*w-9,1))
    if torch.cuda.is_available():
        A = A.cuda()
        b = b.cuda()
    
    return