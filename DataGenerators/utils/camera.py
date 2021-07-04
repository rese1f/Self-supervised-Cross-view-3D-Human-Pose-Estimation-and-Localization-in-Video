# this file is to 

import numpy as np

def get_center(data_3d_std):
    '''
    Get the center of all the users

    Get the center coordinates of all the humans in the plot. Note that the center point is on the horizontal
    plane, so it has a z-value of 0.

    Args:
        data: the 3-D coordinates of the raw data, i.e. data_cluster, data_3d, all of size [n, x, 17, 3]

    Returns:
        datas: of the same size as input data, i.e. [n, x, 17, 3]

    Raises:
        NOError: no error occurred up to now
    '''

    center = np.array([np.sum(data_3d_std[:, 0, 10, 0]), np.sum(data_3d_std[:, 0, 10, 1]), 0]) / 3
    return center


def get_endpoint(center, distance, height, cross):
    '''
    input: distance, cross, center location, height
    output: two endpoint location
    '''
    height = np.random.randint(height[0], height[1], 1)

    start_angle = np.random.random()*2*np.pi

    start_distance = np.random.randint(distance[0],distance[1],1)
    start = center + np.array([start_distance*np.cos(start_angle), start_distance*np.sin(start_angle), height]).reshape(-1)

    if cross:
        end_angle = np.random.random()*2*np.pi
    else:
        end_angle = start_angle + (np.random.random()-0.5)/np.pi/4

    end_distance = np.random.randint(distance[0],distance[1],1)
    end = center + np.array([end_distance*np.cos(end_angle), end_distance*np.sin(end_angle), height]).reshape(-1)

    return start, end

def T(endpoint, frame):
    '''
    input: two endpoint location, frame
    output: location of camera by the time
    '''
    T_mat = np.linspace(endpoint[0], endpoint[1], frame)
    return T_mat

    
def generate_exmat(T_mat, center, tracking):
    '''
    input: location of camera by the time, tracking, center location
    output: exmat of camera by the time
    '''


def w2c(data_3d_std, camera_metadata, frame):
    '''
    input: camera_metadata, data_3d_std
    output: data_c_std
    '''
    center = get_center(data_3d_std)
    print(center)

    distance = camera_metadata['distance']
    height = camera_metadata['height']
    cross = camera_metadata['cross']
    tracking = camera_metadata['tracking']

    endpoint = get_endpoint(center, distance, height, cross)
    
    T_mat = T(endpoint, frame)
    exmat = generate_exmat(T_mat, center, tracking)

    data_c_std = None

    return data_c_std
    

def c2s(data_c_std, inmat):
    '''
    input: w2c, camera_inmat
    output: data_2d_std
    '''
    
    data_2d_std = None
    
    return data_2d_std