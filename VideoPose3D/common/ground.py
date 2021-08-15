import numpy as np

def ground_computer(foot):
    return [single_ground_computer(foot[i]) for i in range(foot.shape[0])]

def single_ground_computer(single_foot):
    o = np.ones((single_foot.shape[0],1))
    A = np.hstack((single_foot[:,:2],o))
    b = single_foot[:,2]
    A_T = A.T
    A_2 = np.linalg.inv(np.matmul(A_T,A))
    ground = np.matmul(np.matmul(A_2, A_T),b)
    # z = ax + by + c
    # ground [a, b, c]
    return ground