import torch
from torch import mean, pow, mul
from torch.nn import parameter

def Regressor(pose_cf, pose_2df, camera, updata, w):
    """
    input
        pose_cf - [1,x,17,3]
        pose_2df - [1,x,17,2]
        camera - [cx,cy,fx,fy]
    return 
        T - [x,3]
        loss - float32
    """
    # [f，17，3]
    pose_cf = pose_cf.squeeze(0)
    pose_2df = pose_2df.squeeze(0)
    # frame
    f = pose_cf.shape[0]
    T = [   for i in range(w,f-w)]


def ABo1f(pose_c, pose_2d, camera):
    """
    give the pose info and camera
    return the two matirx for Ax = B
    """ 
    pose_c = pose_c.transpose(0,1)
    pose_2d = pose_2d.transpose(0,1)
    px = (pose_2d[0]-camera[0])/(camera[2])
    py = (pose_2d[1]-camera[1])/(camera[3])
    pX = pose_c[0]
    pY = pose_c[1]
    pZ = pose_c[2]

    A = torch.tensor([[1,        0,        -mean(px)],
                      [0,        1,        -mean(py)],
                      [mean(px), mean(py), -mean(pow(px,2))-mean(pow(py,2))]])

    B = torch.tensor([[mean(mul(px,pZ))-mean(pX)],
                      [mean(mul(py,pZ))-mean(pY)],
                      [mean(mul(pow(px,2)+pow(py,2),pZ))-mean(mul(px,pX))-mean(mul(py,pY))]])

    return A,B



# for temp

def regressor(pose_cf, pose_2df, camera, updata):
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
    if torch.cuda.is_available():
        Tf = Tf.cuda()
    
    for f in range(frame):
        pose_c = pose_cf[f]
        pose_2d = pose_2df[f]
        T, loss = regressorof(pose_c, pose_2d, camera, updata)
        Tf[f] = T
        lossf += loss
    loss = lossf/frame

    return Tf, loss


def regressorof(pose_c, pose_2d, camera, update):
    """
    regressor of single frame
    camera - [cx,cy,fx,fy]
    """
    pose_c = pose_c.transpose(0,1)
    pose_2d = pose_2d.transpose(0,1)
    # for regression
    px = (pose_2d[0]-camera[0])/(camera[2])
    py = (pose_2d[1]-camera[1])/(camera[3])
    pX = pose_c[0]
    pY = pose_c[1]
    pZ = pose_c[2]

    x = mean(px)
    y = mean(py)
    x2 = mean(pow(px,2))
    y2 = mean(pow(py,2))
    X = mean(pX)
    Y = mean(pY)
    xZ = mean(mul(px,pZ))
    yZ = mean(mul(py,pZ))
    x2y2Z = mean(mul(pow(px,2)+pow(py,2),pZ))
    xX = mean(mul(px,pX))
    yY = mean(mul(py,pY))
    
    parameter_matrix = torch.tensor([[1, 0, -x],
                                     [0, 1, -y],
                                     [x, y, -x2-y2]])

    result_vector = torch.tensor([[xZ-X],
                                  [yZ-Y],
                                  [x2y2Z-xX-yY]])


    T = torch.mm(torch.inverse(parameter_matrix),result_vector).flatten()
    
    # for loss computing
    
    loss = 0
    
    if update:
        J = pose_2d.shape[1]
        alpha = T[0]
        beta = T[1] 
        gamma= T[2]

        loss = mean(pow(-px*gamma+pX-mul(px,pX)+alpha,2)+pow(-py*gamma+pY-mul(py,pY)+beta,2))

    return T, loss