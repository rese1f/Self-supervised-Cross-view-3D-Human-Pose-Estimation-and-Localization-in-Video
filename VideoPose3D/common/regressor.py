import torch
from torch import mean, pow, mul

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

    for f in range(frame):
        pose_c = pose_cf[f]
        pose_2d = pose_2df[f]
        T, loss = regressorof(pose_c, pose_2d, camera, updata)
        Tf[f] = T
        lossf += loss
    loss = lossf/frame
    if torch.cuda.is_available():
        Tf = Tf.cuda()
    return Tf, loss


def regressorof(pose_c, pose_2d, camera, update):
    """
    regressor of single frame
    """
    pose_c = pose_c.transpose(0,1)
    pose_2d = pose_2d.transpose(0,1)
    # for regression
    px = (pose_2d[0]-camera[0]/1000)/(camera[2]/1000)
    py = (pose_2d[1]-camera[1]/1000)/(camera[3]/1000)
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

        X_xZ2 = mean(pow(pX-mul(px,pZ),2))
        X_xZ = mean(pX-mul(px,pZ))
        Y_yZ2 = mean(pow(pY-mul(py,pZ),2))
        Y_yZ = mean(pY-mul(py,pZ))
        x2Z = mean(mul(pow(px,2),pZ))
        y2Z = mean(mul(pow(py,2),pZ))

        loss = alpha**2/J + beta**2/J +(x2+y2)*gamma**2 + X_xZ2 + Y_yZ2 -2*x*alpha*gamma -2*y*beta*gamma + 2*X_xZ*alpha + 2*Y_yZ*beta - 2*(xX+yY-x2Z-y2Z)*gamma 
        



    return T, loss