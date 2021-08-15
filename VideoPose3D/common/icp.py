import numpy as np
from numpy.core.fromnumeric import squeeze
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
    info_shape = A.shape; cloudshape = (info_shape[0],info_shape[1]*info_shape[2],info_shape[3])
    centroid_A = torch.mean(torch.mean(A, dim=2),dim=1)
    centroid_B = torch.mean(torch.mean(B, dim=2),dim=1)
    AA = high_dim_sub(A, centroid_A).reshape(cloudshape);    BB = high_dim_sub(B, centroid_B).reshape(cloudshape)

    # rotation matrix
    W = torch.matmul(BB.transpose(1,2), AA)
    U, s, VT = torch.linalg.svd(W)
    R = torch.matmul(U, VT)

    # special reflection case
    #if torch.linalg.det(R) < 0:
    #   VT[:,:,2,:] = torch.mul(VT[:,:,2,:], torch.tensor(-1))
    #   R = torch.matmul(U, VT)


    # translation
    shape = centroid_A.shape
    t = centroid_B.reshape(shape[0],shape[1],1) - torch.matmul(R,centroid_A.reshape(shape[0],shape[1],1))
    # homogeneous transformation
    T = torch.zeros((info_shape[0],4,4)); T[:,:] = torch.eye(4)
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = t.squeeze()

    #print(T)
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
    distance = torch.sum(torch.sum(torch.norm(torch.sub(A,B),dim=3), dim=2),dim=1)
    return distance

def high_dim_matmul(T,src):
    '''
    Operate high dimention matrix multiplication without standard size
    Input:
        src: [v,n,N,3/4] torch tensor of points
        T: [v,4,4] torch tensor of points
    Output:
        src: [v,n,17,3/4] torch tensor of points
    '''
    shape = src.shape; #print(shape)
    T = T.unsqueeze(1).unsqueeze(2).repeat(1,shape[1],shape[2],1,1).reshape(shape[0]*shape[1]*shape[2],shape[3],shape[3]) # [v,n,N,4,4]
    src = src.unsqueeze(4).reshape(shape[0]*shape[1]*shape[2],shape[3],1).contiguous() # shape of [v,n,N,4,1]
    src_new = torch.matmul(T, src).reshape(shape[0],shape[1],shape[2],shape[3])
    #print(torch.matmul(T, src).reshape(shape[0],shape[1],shape[2],shape[3]).contiguous().shape)
    #print(torch.matmul(T, src).permute(1,2,0,3,4).reshape(shape[0],shape[1],shape[2],shape[3]).shape)
        # [v,n,3/4,3/4] reshape2 [v,n,17,3/4]
    return src_new

def high_dim_matmul_2(T,src):
    '''
    Operate high dimention matrix multiplication without standard size
    Input:
        src: [v,n,N,3/4] torch tensor of points
        T: [v,n,4,4] torch tensor of points
    Output:
        src: [v,n,17,3/4] torch tensor of points
    '''

    shape = src.shape; 
    T = T.unsqueeze(1).unsqueeze(2).repeat(1,shape[1],shape[2],1,1).reshape(shape[0]*shape[1]*shape[2],3,4) # [v,n,N,4,4]
    src = src.unsqueeze(4).reshape(shape[0]*shape[1]*shape[2],shape[3],1).contiguous() # shape of [v,n,N,4,1]
    src_new = torch.matmul(T, src).reshape(shape[0],shape[1],shape[2],shape[3]-1)
        # [v,n,3/4,3/4] reshape2 [v,n,17,3/4]
    return src_new

def high_dim_sub(A,B):
    '''
    Operate high dimention tensor subtraction without standard size
    Input:
        src: [v,n,N,3] torch tensor of points
        T: [v,n,3] torch tensor of points
    Output:
        src: [v,n,17,3/4] torch tensor of points
    '''
    A = A.permute(1,2,0,3) # shape of [N,v,n,3/4,1]
    A = torch.sub(A, B).permute(2,0,1,3)
        # [v,n,3/4,3/4] reshape2 [v,n,17,3/4]
    return A

def icp(A, B, init_pose=None, max_iterations=100, tolerance=0.001):
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
    #print(src.shape);print(A.shape)
    src[:,:,:,0:3].copy_(A);    dst[:,:,:,0:3].copy_(B); A_h.copy_(src)

    # apply the initial pose estimation; 
    if init_pose is not None:
        src = torch.matmul(init_pose, src)

    prev_error = 0 # error
    T_0 = torch.zeros((A.shape[0],4,4)); T_0[:] = torch.eye(4) # [v,n,4,4]
    R = torch.zeros((A.shape[0],A.shape[1],3,3)) # [v,n,3,3]
    t = torch.zeros((A.shape[0],A.shape[1],3,1)) # [v,n,3,1]
    loss_list = 0 # total loss
    T_list = list(); T_list.append(T_0)
    #print(T)

    for i in range(max_iterations):
        # find the nearest neighbours between the current source and destination points
        distances = euc_dist(src[:,:,:,0:3], dst[:,:,:,0:3]) # [v,1]

        # compute the transformation between the current source and nearest destination points
        T_temp,R_temp,t_temp = best_fit_transform(src[:,:,:,0:3], dst[:,:,:,0:3]) # [v,4,4]

        # update the current source
        # refer to "Introduction to Robotics" Chapter2 P28. Spatial description and transformations
        #print(src.shape); print(T_temp.shape)
        src = high_dim_matmul(T_temp,src) # [v,n,N,4]

        # check error
        mean_error = torch.sum(distances) / distances.shape[0] # float
        T_list.append(torch.matmul(T_temp,T_list[-1])) # [v,n,4,4]
        if abs(prev_error-mean_error) < tolerance or i == max_iterations-1:
            R = R_temp; t = t_temp
            #T = torch.mean(T_list[-1], dim=1)
            T = T_list[-1]
            loss = euc_dist(high_dim_matmul_2(T[:,0:3,:],A_h),B) # [v,1]
            
        prev_error = mean_error # float

    # calculcate final tranformation
    #T,_,_ = best_fit_transform(A, src[0:3,:].T)
    
    T = T_list[-1].unsqueeze(0); loss = loss.unsqueeze(0)

    #print(T)
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

    data_shape = data_pos_3d.shape # get the shape of the tensor
    #T = torch.zeros((data_shape[2],sum(range(data_shape[0])),4,4)) # homogeneous transformation matrix of [x,sum(range(v)),4,4]
    #loss = torch.zeros((data_shape[3],sum(range(data_shape[0])))) # loss of [x,sum(range(v)),1]
    
    data_pos_3d = data_pos_3d.permute(3,0,1,2,4) # [skeleton, view, person, frame, 3]
    data_pos_3d = data_pos_3d + data_trans # adding translational vector
    #data_pos_3d = data_pos_3d.permute(3,1,2,0,4) # [frame, view, person, skeleton, 3]
    data_pos_3d = data_pos_3d.permute(1,2,3,0,4) # [view, person, frame, skeleton, 3]
    #data_pos_3d.reshape(data_shape[1], data_shape[0] * data_shape[2], data_shape[3])

    # combinations of view
    '''
    data_combd_A = [];    data_combd_B = []
    for j in range(data_shape[2]):
        comb_list = list(itertools.combinations(data_pos_3d[j], 2))
        for i in range(len(comb_list)):
            if i == 0: 
                data_A = comb_list[0][0].reshape(1,data_shape[1],data_shape[3],data_shape[4])
                data_B = comb_list[0][1].reshape(1,data_shape[1],data_shape[3],data_shape[4])
            else:
                data_A = torch.cat((data_A,comb_list[i][0].reshape(1,data_shape[1],data_shape[3],data_shape[4])),dim=0)
                data_B = torch.cat((data_B,comb_list[i][1].reshape(1,data_shape[1],data_shape[3],data_shape[4])),dim=0)
                # [v,n,17,3]
        data_combd_A.append(data_A); data_combd_B.append(data_B)
    #print(len(data_combd_A))
    #print(len(data_combd_B))
    '''
    # generate the combinations of different views, and reswap the dimention between frame and dimention
    data_combd_A,data_combd_B = unzip([i[0].unsqueeze(0), i[1].unsqueeze(0)] for i in itertools.combinations(data_pos_3d, 2) )
    data_combd_A = data_combd_A.permute(2,0,1,3,4); data_combd_B = data_combd_B.permute(2,0,1,3,4)

    # calculate the transformation matirx and loss
    T,loss = unzip([icp(i[0], i[1]) for i in zip(data_combd_A,data_combd_B)])


    return T, loss

def unzip(list):
    """unzip the tensor tuple list

    Args:
        list: contains tuple of segemented tensors
    """
    A, B = zip(*list)
    A = torch.cat(A)
    B = torch.cat(B)
    return A, B


if __name__ == "__main__":
    
    data = np.load("output\data_output.npz",allow_pickle=True)
    
    a = torch.tensor(data["pose_pred"])
    b = torch.tensor(data["T"])
    
    print(a.shape); 
    print(b.shape)
    
    #a = torch.rand((2,3,200,17,3))    # 20 points for test
    #b = torch.rand((2,3,200,3))
    
    T, distances = icp_multi(a, b)
    #print(T.shape)
    print(distances)
    
    #T = T.permute(1,0,2,3)
    
    a = a.permute(3,0,1,2,4) # [skeleton, view, person, frame, 3]
    a = a + b # adding translational vector
    #data_pos_3d = data_pos_3d.permute(3,1,2,0,4) # [frame, view, person, skeleton, 3]
    a = a.permute(1,2,3,0,4) # [view, person, frame, skeleton, 3]
    pos_sample = torch.ones((a.shape[1],a.shape[2],a.shape[3],4))
    pos_sample[:,:,:,0:3].copy_(a[0]) # [person, frame, skeleton, 4]
    pos_sample = pos_sample.unsqueeze(4).reshape(a.shape[1],a.shape[2],a.shape[3],4,1).permute(2, 0, 1, 3, 4) # [skeleton, person, frame, 4, 1]
    #print(pos_sample.shape)

    a1 = a[0]; a2 = [1]
    T_sample = T.permute(1,0,2,3)[0,:,0:3,:] # [view0, frame, 3, 4]
    #print(T_sample.shape)

    pos_turth = a[1]
    pos_transform = torch.matmul(T_sample,pos_sample).permute(1,2,0,3,4).reshape(a.shape[1],a.shape[2],a.shape[3],3)

    print(pos_transform.shape)
    print(pos_turth.shape)

    #print(pos_turth - pos_transform)
    np.savez_compressed("output\data_icp_comp.npz",truth=pos_turth.cpu().numpy(),transform=pos_transform.cpu().numpy())
    #np.savez_compressed("output\data_icp_comp.npz",truth=a[1].cpu().numpy(),transform=a[0].cpu().numpy())