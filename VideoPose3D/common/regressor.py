import torch
from torch import mean, pow, mul

def Regressor(pose_cf, pose_2df, camera, update, w):
    """
    input
        pose_cf - [1,x,17,3]
        pose_2df - [1,x,17,2]
        camera - [cx,cy,fx,fy]
    return 
        T - [x,3]
        loss - float32
    """
    # [f，3, 17]
    pose_cf = pose_cf.squeeze(0).transpose(1,2)
    pose_2df = pose_2df.squeeze(0).transpose(1,2)
    # frame
    f = pose_cf.shape[0]
    # base is to calculate all the A B matrix for single frame
    # base - list(matrix A, matrix B), len = f
    base = ABof(pose_cf,pose_2df,camera,f)
    # T - [f-2w,3]
    T = torch.stack([solver(base[i-w:i+w+1],w) for i in range(w,f-w)]).unsqueeze(-1)
    if torch.cuda.is_available():
        T = T.cuda()
    loss = 0
    if update:
        loss = loss_total(pose_cf, pose_2df, camera, T, w)

    return T,loss


def ABof(pose_cw, pose_2dw, camera, f):
    """
    pose_c: [f,3,17]
    """
    return [ABo1f(pose_cw[i],pose_2dw[i],camera) for i in range(f)]


def ABo1f(pose_c, pose_2d, camera):
    """
    pose_c: [3,17]
    give the pose info and camera
    return the two matirx for Ax = B
    """
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


def solver(matrix_list,w):
    """
    input: list[(A,B),(A,B),...]
    """
    A = sum([matrix_list[j][0]*(0.5**abs(w-j)) for j in range(len(matrix_list))])
    B = sum([matrix_list[j][1]*(0.5**abs(w-j)) for j in range(len(matrix_list))])
    T = torch.linalg.solve(A,B)

    return T


def loss_total(pose_cf, pose_2df, camera, T, w):
    """
    computing loss for regressor to backward the network
    pose_cf - [f,3,17]
    psoe_2df -  [f,2,17]
    camera - [cx,cy,fx,fy]
    T - [f-2w,3]
    w - int
    """
    # p: [f,17]
    px = (pose_2df[:,0]-camera[0])/(camera[2])
    py = (pose_2df[:,1]-camera[1])/(camera[3])
    pX = pose_cf[:,0]
    pY = pose_cf[:,1]
    pZ = pose_cf[:,2]
    # total weight W to divide
    W = 4*(1-0.5**(w+1))-1
    # p[i:i+2*w+1]
    loss = sum([loss_owf(px[i:i+2*w+1],py[i:i+2*w+1],pX[i:i+2*w+1],pY[i:i+2*w+1],pZ[i:i+2*w+1],T[i],w) for i in range(T.shape[0])])/W/T.shape[0]
    return loss


def loss_owf(px, py, 
             pX, pY, pZ, 
             T, 
             w):
    """
    loss of w frame of one group of weight
    x,y,X,Y,Z - [2w+1,17]
    T - (alpha,beta,gamma)
    """
    loss = sum([loss_o1f(px[i],py[i],pX[i],pY[i],pZ[i],T)*(0.5**abs(w-i)) for i in range(2*w+1)])
    return loss


def loss_o1f(px, py, 
             pX, pY, pZ, 
             T):
    """
    loss of single frame
    x,y,X,Y,Z - [17]
    T - (alpha,beta,gamma)
    """
    alpha = T[0]
    beta = T[1]
    gamma = T[2]
    loss = mean(pow(-px*gamma+pX-mul(px,pZ)+alpha,2)+pow(-py*gamma+pY-mul(py,pZ)+beta,2))
    
    return loss
