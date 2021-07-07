import torch

def regressor(pose_cf, pose_2df):
    """
    input
        pose_cf - [1,x,17,3]
        pose_2df - [1,x,17,2]
    return 
        T - [x,3]
        loss
    """
    f = pose_cf.shape[1]
    Tf = torch.zeros((f,3))
    lossf = 0
    for frame in range(f):
        pose_c = pose_cf[:,f]
        pose_2d = pose_2df[:,f]
        T, loss = regressorof(pose_c, pose_2d)
        Tf[f] = T
        lossf += loss
    loss = lossf/f
    
    print(Tf)


def regressorof(pose_c, pose_2d):
    """
    regressor of single frame
    """
    T = torch.tensor([0,0,0])
    loss = 0
    return T, loss
