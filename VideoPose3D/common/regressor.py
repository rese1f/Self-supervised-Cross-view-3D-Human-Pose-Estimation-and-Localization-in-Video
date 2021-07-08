import torch

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
    return Tf, loss


def regressorof(pose_c, pose_2d, camera):
    """
    regressor of single frame
    """
    loss = 0
    pose_2d[:,0] = (pose_2d[:,0]-camera[0])/(camera[2]*1000)
    pose_2d[:,1] = (pose_2d[:,1]-camera[1])/(camera[3]*1000)
    x = torch.mean(pose_2d[:,0])
    y = torch.mean(pose_2d[:,1])
    x2 = torch.mean(torch.pow(pose_2d,2)[:,0])
    y2 = torch.mean(torch.pow(pose_2d,2)[:,1])
    x2y2 = x2 + y2
    xZ = torch.mean(torch.mul(pose_2d[:,0], pose_c[:,-1]))
    yZ = torch.mean(torch.mul(pose_2d[:,1], pose_c[:,-1]))
    X = torch.mean(pose_c[:,0])
    Y = torch.mean(pose_c[:,1])
    x2y2Z = torch.mean(torch.mul(torch.pow(pose_2d[:,0], 2), pose_c[:,-1])+torch.mul(torch.pow(pose_2d[:,1], 2), pose_c[:,-1]))
    xX = torch.mean(torch.mul(pose_2d[:,0], pose_c[:,0]))
    yY = torch.mean(torch.mul(pose_2d[:,1], pose_c[:,1]))
    parameter_matrix = torch.tensor([[1, 0, -x],
                                     [0, 1, -y],
                                     [x, y, -x2y2]])
    result_vector = torch.tensor([[xZ-X],
                                  [yZ-Y],
                                  [x2y2Z-xX-yY]])
    T = torch.mm(torch.inverse(parameter_matrix),result_vector).flatten()
    
    return T, loss

