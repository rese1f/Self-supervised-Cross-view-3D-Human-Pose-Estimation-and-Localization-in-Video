import numpy as np
import torch
import itertools

def best_fit_transform(A,B):
    '''
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: [v,n,N,3] numpy array of corresponding 3D points
      B: [v,n,N,3] numpy array of corresponding 3D points
    Returns:
      T: [v,n,4,4] homogeneous transformation matrix
      R: [v,n,3,3] rotation matrix
      t: [v,n,3] column vector
    '''

    assert len(A) == len(B)

    # translate points to their centroids
    info_shape = A.shape
    centroid_A = torch.mean(A, dim=2)
    centroid_B = torch.mean(B, dim=2)
    AA = A - centroid_A;    BB = B - centroid_B

    # rotation matrix
    W = torch.matmul(BB.transpose(2,3), AA)
    U, s, VT = torch.linalg.svd(W)
    R = torch.matmul(U, VT)

    # special reflection case
    if torch.linalg.det(R) < 0:
       VT[:,:,2,:] = torch.mul(VT[:,:,2,:], torch.tensor(-1))
       R = torch.matmul(U, VT)


    # translation
    shape = centroid_A.shape
    t = centroid_B.reshape(shape[0],shape[1],shape[2],1) - torch.matmul(R,centroid_A.reshape(shape[0],shape[1],shape[2],1))

    # homogeneous transformation
    T = torch.zeros((info_shape[0],info_shape[1],4,4)); T[:,:] = torch.eye(4)
    T[:, :, 0:3, 0:3] = R
    T[:, :, 0:3, 3] = t

    return T, R, t

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nx3 array of points
        dst: Nx3 array of points
    Output:
        distances: Euclidean distances (errors) of the nearest neighbor
        indecies: dst indecies of the nearest neighbor
    '''

    indecies = np.zeros(src.shape[0], dtype=np.int)
    distances = np.zeros(src.shape[0])
    for i, s in enumerate(src):
        min_dist = np.inf
        for j, d in enumerate(dst):
            dist = np.linalg.norm(s-d)
            if dist < min_dist:
                min_dist = dist
                indecies[i] = j
                distances[i] = dist    
    return distances, indecies

def euc_dist(A,B):
    '''
    Find the nearest (Euclidean) distance for A and B
    Input:
        src: [v,n,17,3] torch tensor of points
        dst: [v,n,17,3] torch tensor of points
    Output:
        distances: Euclidean distances (loss) of the point A and B: [v,1]
        
    '''
    distance = torch.mean(torch.mean(torch.norm(torch.sub(A,B),dim=3), dim=2),dim=1)
    return distance

def high_dim_matmul(T,src):
    '''
    Operate high dimention matrix multiplication without standard size
    Input:
        src: [v,n,N,3/4] torch tensor of points
        T: [v,n,4,4] torch tensor of points
    Output:
        src: [v,n,17,3/4] torch tensor of points
    '''
    src_ = src.reshape(src.shape[0],src.shape[1],src.shape[2],src.shape[3],1).permute(2,0,1,3,4) # shape of [N,v,n,3/4,1]
    src = (torch.matmul(T, src_).permute(1,2,0,3).reshape(src.shape[0],src.shape[1],src.shape[2],src.shape[3])) 
        # [v,n,3/4,3/4] reshape2 [v,n,17,3/4]
    return src

def icp(A, B, init_pose=None, max_iterations=50, tolerance=0.001):
    '''
    The Iterative Closest Point method
    Input:
        A: [v,n,17,3] torch tensor of source 3D points
        B: [v,n,17,3] torch tensor of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation
        distances: Euclidean distances (errors) of the nearest neighbor
    '''

    # make points homogeneous, copy them so as to maintain the originals
    A_h = torch.ones((A.shape[0],A.shape[1],A.shape[2],4)) # [v,n,N,4]
    src = torch.ones((A.shape[0],A.shape[1],A.shape[2],4)) # [v,n,N,4]
    dst = torch.ones((B.shape[0],B.shape[1],B.shape[2],4)) # [v,n,N,4]
    src[:,:,:,0:3].copy_(A);    dst[:,:,:,0:3].copy_(B); A_h.copy_(src)

    # apply the initial pose estimation; 
    if init_pose is not None:
        src = torch.matmul(init_pose, src)

    prev_error = 0 # error
    T = torch.zeros((A.shape[0],A.shape[1],4,4)); T[:,:] = torch.eye(4) # [v,n,4,4]
    R = torch.zeros((A.shape[0],A.shape[1],3,3)) # [v,n,3,3]
    t = torch.zeros((A.shape[0],A.shape[1],3,1)) # [v,n,3,1]
    loss = 0 # total loss

    for i in range(max_iterations):
        # find the nearest neighbours between the current source and destination points
        distances = euc_dist(src[:,:,:,0:3], dst[:,:,:,0:3]) # [v,1]

        # compute the transformation between the current source and nearest destination points
        T_temp,R_temp,t_temp = best_fit_transform(src[:,:,:,0:3], dst[:,:,:,0:3]) # [v,n,4,4]

        # update the current source
        # refer to "Introduction to Robotics" Chapter2 P28. Spatial description and transformations
        src = high_dim_matmul(T_temp,src) # [v,n,N,4]

        # check error
        mean_error = torch.sum(distances) / distances.shape[0] # float
        T = torch.matmul(T_temp,T) # [v,n,4,4]
        if abs(prev_error-mean_error) < tolerance or i == max_iterations-1:
            R = R_temp; t = t_temp
            loss = euc_dist(high_dim_matmul(T[:,:,0:3,:],A_h),B) # [v,1]
            break
        prev_error = mean_error # float

    # calculcate final tranformation
    #T,_,_ = best_fit_transform(A, src[0:3,:].T)
    

    return T, loss
    
def icp_multi(data_pos_3d, data_trans, init_pose=None, max_iterations=50, tolerance=0.001):
    '''
    The Iterative Closest Point method 4 multiframes
    Input:
        data_pos_3d: [v,n,x,17,3] torch tensor of source 3D points
        data_trans: [v,n,x,3] numpy array of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation
        loss: Euclidean distances (errors) of the tranformation
    '''

    data_shape = data_pos_3d.shape
    T = torch.zeros((data_shape[3],sum(range(data_shape[0])),4,4)) # homogeneous transformation matrix of [x,sum(range(v)),4,4]
    loss = torch.zeros((data_shape[3],sum(range(data_shape[0])),1)) # loss of [x,sum(range(v)),1]
    
    data_pos_3d = data_pos_3d.permute(3,0,1,2,4)
    data_pos_3d = data_pos_3d + data_trans
    data_pos_3d = data_pos_3d.permute(3,1,2,0,4) # [frame, view, person, skeleton, 3]
    #data_pos_3d.reshape(data_shape[1], data_shape[0] * data_shape[2], data_shape[3])

    # combinations of view

    for i in range(data_shape[1]):
        comb_list = list(itertools.combinations(data_pos_3d[i], 2))
        for i in range(len(comb_list)):
            if i == 0: 
                data_A = comb_list[0,0].reshape(1,data_shape[1],data_shape[3],data_shape[4])
                data_B = comb_list[0,1].reshape(1,data_shape[1],data_shape[3],data_shape[4])
            else:
                data_A = torch.cat((data_A,comb_list[i,0].reshape(1,data_shape[1],data_shape[3],data_shape[4])),dim=0)
                data_B = torch.cat((data_B,comb_list[i,1].reshape(1,data_shape[1],data_shape[3],data_shape[4])),dim=0)
                # [v,n,17,3]

        T[i],loss[i] = icp(data_A, data_B)

    return T, loss

if __name__ == "__main__":
    A = np.random.randint(0,101,(20,3))    # 20 points for test
    
    rotz = lambda theta: np.array([[np.cos(theta),-np.sin(theta),0],
                                       [np.sin(theta),np.cos(theta),0],
                                       [0,0,1]])
    trans = np.array([2.12,-0.2,1.3])
    B = A.dot(rotz(np.pi/4).T) + trans
    

    print(B)
    print(A)
    #T, distances = icp(A, B)

    #np.set_printoptions(precision=3,suppress=True)
    #print T