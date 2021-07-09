import torch
from torch import mean, pow, mul

def regressor(pose_cf, pose_2df, camera):
    """
    input
        pose_cf - [1,x,17,3]
        pose_2df - [1,x,17,2]
        camera - [cx,cy,fx,fy]
    return 
        Tf - [x,3]
        loss - float32
    """
    pose_cf = pose_cf.squeeze(0)
    pose_2df = pose_2df.squeeze(0)
    frame = pose_cf.shape[0]
    Tf = torch.zeros((frame,3))
    lossf = 0

    for f in range(frame):
        pose_c = pose_cf[f]
        pose_2d = pose_2df[f]
        T, loss = regressorof(pose_c, pose_2d, camera)
        Tf[f] = T
        lossf += loss
    loss = lossf/frame
    if torch.cuda.is_available():
        Tf = Tf.cuda()
    return Tf, loss


def regressorof(pose_c, pose_2d, camera):
    """
    regressor of single frame
    """
    pose_c = pose_c.transpose(0,1)
    pose_2d = pose_2d.transpose(0,1)
    # for regression
    p_x = (pose_2d[0]-camera[0]/1000)/(camera[2]/1000)
    p_y = (pose_2d[1]-camera[1]/1000)/(camera[3]/1000)
    p_X = pose_c[0]
    p_Y = pose_c[1]
    p_Z = pose_c[2]

    x = mean(p_x)
    y = mean(p_y)
    x2 = mean(pow(p_x,2))
    y2 = mean(pow(p_y,2))
    X = mean(p_X)
    Y = mean(p_Y)
    xZ = mean(mul(p_x,p_Z))
    yZ = mean(mul(p_y,p_Z))
    x2y2Z = mean(mul(pow(p_x,2)+pow(p_y,2),p_Z))
    xX = mean(mul(p_x,p_X))
    yY = mean(mul(p_y,p_Y))
    
    parameter_matrix = torch.tensor([[1, 0, -x],
                                     [0, 1, -y],
                                     [x, y, -x2-y2]])

    result_vector = torch.tensor([[xZ-X],
                                  [yZ-Y],
                                  [x2y2Z-xX-yY]])

    T = torch.mm(torch.inverse(parameter_matrix),result_vector).flatten()
    
    # for loss computing
    loss = 0
    J = pose_2d.shape[1]
    alpha = T[0]
    beta = T[1]
    gamma= T[2]

    return T, loss