
# How to check grad?
# Hint: torch.sum(tensor).backward(retain_graph=True)

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
    T,loss = zip(*[unzip([subregressor(i[0][j], i[1][j], update) for i in zip(pose_pred,pose_2d)]) for j in range(N)])
    T = torch.stack(T)
    mean_loss = torch.stack(loss).mean()
    if update:
        mean_loss.backward(retain_graph=True)
    return T, mean_loss

def unzip(list):
    """unzip the tensor tuple list

    Args:
        list: contains tuple of segemented tensors
    """
    T, loss = zip(*list)
    T = torch.cat(T)
    mean_loss = torch.cat(loss).mean()
    return T, mean_loss

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

    # get component [128,17]
    x = pose_2d[...,0]
    y = pose_2d[...,1]
    xzX = torch.mul(x,pose_pred[...,2]).add(pose_pred[...,0])
    yzY = torch.mul(y,pose_pred[...,2]).add(pose_pred[...,1])
    
    A_tuple, b_tuple = zip(*[submatrix(w, i, x[i], y[i], xzX[i], yzY[i]) for i in range(w)])
    A_mse = torch.cat(A_tuple)
    b_mse = torch.cat(b_tuple)
    
    # first difference
    lambda_1 = 1e-3
    I_1 = torch.eye((3*w-3), device=device) * lambda_1
    O = torch.zeros((3*w-3,3), device=device)
    A_1 = torch.cat((I_1,O), dim=1) - torch.cat((O,I_1), dim=1)
    b_1 = torch.zeros((3*w-3,1), device=device)

    A = torch.cat((A_mse,A_1), dim=0)
    b = torch.cat((b_mse,b_1), dim=0)
    T = torch.mm(torch.mm(torch.inverse(torch.mm(A.T,A)),A.T),b)
    loss = 0
    if update:
        loss = torch.mm((torch.mm(A,T)-b).T,(torch.mm(A,T)-b))/(2*J*w+3*w-3)**2
    T = T.reshape(w,3)
    return T, loss

def submatrix(w, i, x, y, xzX, yzY):
    """create a matrix block

    Args:
        x (tensor): [17]
        y (tensor): [17]
        xzX (tensor): [17]
        yzY (tensor): [17]
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    J = x.shape[0]

    fill_0 = torch.zeros((J,1), device=device, requires_grad=True)
    fill_1 = -torch.ones((J,1), device=device, requires_grad=True)
    fill_x = torch.cat((fill_1, fill_0, x.unsqueeze(-1)), dim=1)
    fill_y = torch.cat((fill_0, fill_1, y.unsqueeze(-1)), dim=1)
    fill = torch.cat((fill_x, fill_y), dim=0)
    zero_l = torch.zeros((J*2,3*i), device=device, requires_grad=True)
    zero_r = torch.zeros((J*2,3*w-3*i-3), device=device, requires_grad=True)
    
    Ai = torch.cat((zero_l, fill, zero_r), dim=1)
    bi = torch.cat((xzX.unsqueeze(-1), yzY.unsqueeze(-1)), dim=0)

    return Ai, bi