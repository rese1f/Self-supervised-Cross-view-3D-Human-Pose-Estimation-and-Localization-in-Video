import torch
from torch.nn import parameter

def regressor(pose_cf, pose_2df):
    """
    input
        pose_cf - [1,x,17,3]
        pose_2df - [1,x,17,2]
    return 
        T - [x,3]
        loss
    """
    frame = pose_cf.shape[1]
    Tf = torch.zeros((frame,3))
    lossf = 0
    for f in range(frame):
        pose_c = pose_cf[:,f]
        pose_2d = pose_2df[:,f]
        T, loss = regressorof(pose_c, pose_2d)
        Tf[f] = T
        lossf += loss
    loss = lossf/frame
    
    return Tf, loss


def regressorof(pose_c, pose_2d):
    """
    regressor of single frame
    """

    loss = 0
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
    T = torch.mm(torch.inverse(parameter_matrix),result_vector)
    print(T)
    return T, loss



pose_2d=torch.tensor([[2272.1101, 2763.8066],
        [2256.8547, 2746.6528],
        [2421.2485, 2746.6279],
        [2546.9055, 2663.6597],
        [2287.1313, 2780.6968],
        [2398.6116, 2693.3252],
        [2361.9680, 2499.3904],
        [2188.9670, 2765.5984],
        [2102.9236, 2777.0093],
        [2082.8821, 2808.4573],
        [2039.5996, 2788.6091],
        [2137.1245, 2803.8896],
        [2245.0320, 2836.3259],
        [2299.2070, 2870.0930],
        [2114.9277, 2747.8757],
        [2219.2788, 2781.4797],
        [2323.7583, 2893.2749]])
pose_c=torch.tensor([[ 5.2104e-06,  7.6789e-05, -5.3667e-05],
        [-1.2964e-01, -1.4869e-01,  6.6221e-03],
        [ 1.6418e-01,  5.5313e-01, -2.2193e-01],
        [ 2.9000e-01,  1.4654e+00, -3.5212e-01],
        [ 1.2966e-01,  1.4865e-01, -6.6056e-03],
        [ 4.2381e-01,  7.8491e-01, -2.6523e-01],
        [ 7.8945e-01,  8.7428e-01, -4.6404e-01],
        [-9.2208e-02, -2.0362e-01, -1.8943e-01],
        [-8.0624e-02, -3.7353e-01, -4.0590e-01],
        [-9.6038e-02, -6.1744e-01, -5.4073e-01],
        [-1.6521e-01, -7.5374e-01, -3.9808e-01],
        [ 1.1464e-02, -2.7332e-01, -4.1548e-01],
        [ 2.0435e-01, -2.4766e-01, -1.0988e+00],
        [-1.4736e-01, -1.0645e+00, -1.2634e+00],
        [-1.5896e-01, -6.1522e-01, -3.9627e-01],
        [-3.5629e-01, -8.5610e-01, -5.6617e-01],
        [ 8.4115e-02, -1.5063e+00, -7.2465e-01]])
pose_gt=torch.tensor([[4003.3018, 6793.0088, 4649.8828],
        [3926.2231, 6688.6572, 4613.8809],
        [4344.1860, 6569.6733, 4531.8560],
        [4749.6084, 6367.4883, 4563.1694],
        [4080.3806, 6897.3599, 4685.8848],
        [4479.4619, 6715.1885, 4746.3584],
        [4707.1392, 6591.5254, 5117.6938],
        [3802.3892, 6893.0884, 4714.6069],
        [3571.4424, 6996.1304, 4760.7861],
        [3465.2578, 7005.6538, 4701.4731],
        [3379.0750, 7042.6821, 4767.8530],
        [3685.8037, 7094.7529, 4770.8306],
        [3944.2046, 7055.2656, 4677.5596],
        [3947.6135, 6875.5308, 4492.6260],
        [3579.2158, 6848.7104, 4721.6821],
        [3771.3416, 6720.0122, 4563.8120],
        [3884.7583, 6710.3247, 4341.6704]])

# regressorof(pose_c, pose_2d)