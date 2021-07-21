
from os import device_encoding
import torch

def regressor(camera, pose_pred, pose_2d, w, update):
    """create a linear regression in single view

    Args:
        camera (tensor): camera_parameters[cx,cy,fx,fy]
        pose_pred (tensor): predicted_3d_pose with shape[number,frame,joint,3]
        pose_2d (tensor): pixel_2d_pose with shape[number,frame,joint,2]
        w (int): number of frames in one shot
        update (bool): if compute loss
    """
    # make a assignment x=(x-c)/f, y=(y-c)/f
    pose_2d[...,0].add_(-camera[0]).mul_(1/camera[2])
    pose_2d[...,1].add_(-camera[1]).mul_(1/camera[3])

    # split via person and width
    N = pose_2d.shape[0]
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
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # initial the matrix
    w = pose_2d.shape[0]
    J = pose_2d.shape[1]
    A = torch.zeros((2*J*w,3*w), device=device)
    b = torch.zeros((2*J*w,1), device=device)

    # get component [128,17]
    x = pose_2d[...,0]
    y = pose_2d[...,1]
    xzX = torch.mul(x,pose_pred[...,2]).add(pose_pred[...,0])
    yzY = torch.mul(y,pose_pred[...,2]).add(pose_pred[...,1])

    [submatrix(A, b, i, x[i], y[i], xzX[i], yzY[i]) for i in range(w)]
    
    # first difference
    lambda_1 = 1e-3
    I_1 = torch.eye((3*w-3), device=device) * lambda_1
    O = torch.zeros((3*w-3,3), device=device)
    A_1 = torch.cat((I_1,O), dim=1) - torch.cat((O,I_1), dim=1)
    b_1 = torch.zeros((3*w-3,1), device=device)

    # second difference
    A = torch.cat((A,A_1), dim=0)
    b = torch.cat((b,b_1), dim=0)
    
    T = torch.mm(torch.mm(torch.inverse(torch.mm(A.T,A)),A.T),b)
    
    loss = 0
    if update:
        loss = torch.mm((torch.mm(A,T)-b).T,(torch.mm(A,T)-b))/(2*J*w+3*w-3)**2
    
    T = T.reshape(w,3)
    
    return T, loss

def submatrix(A, b, i, x, y, xzX, yzY):
    """create a matrix block

    Args:
        x (tensor): [17]
        y (tensor): [17]
        xzX (tensor): [17]
        yzY (tensor): [17]
    """
    J = x.shape[0]
    # Ai - [34,3]
    # bi - [34]
    Ai = torch.zeros((J*2,3))
    bi = torch.zeros((J*2))
    Ai[:J,0] = -1
    Ai[J:,1] = -1
    Ai[:J,2] = x
    Ai[J:,2] = y
    bi[:J] = xzX
    bi[J:] = yzY
    
    A[i*2*J:(i+1)*2*J,i*3:(i+1)*3] = Ai
    b[i*2*J:(i+1)*2*J,:] = bi.unsqueeze_(-1)
        
    return