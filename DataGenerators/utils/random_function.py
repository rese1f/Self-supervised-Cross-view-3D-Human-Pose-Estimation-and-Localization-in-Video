# this file contains some functions to process raw single data

import random
import numpy as np
from numpy.matrixlib.defmatrix import matrix


def extract(dict_keys, min, max):
    '''
    input: dict_keys([...,...])
    output: dict_keys(list with len in [min,max])
    '''
    n = random.randint(min, max+1)
    keys = random.sample(dict_keys, n)
    return keys

def pre_process(array, shift, distance, rotation):
    
    # random shift timeline
    frame = random.randint(0, shift)
    array = array[frame:]

    # random traslate the raw array
    dis = np.random.normal(distance,distance/3,2)
    x_flag, y_flag = random.choice((-1,1)), random.choice((-1,1))
    x, y = dis[0]*x_flag, dis[1]*y_flag
    trans = np.zeros_like(array)
    trans[:,:,0], trans[:,:,1] = x, y
    array += trans

    # random rotation the raw array
    if rotation is False:
        return array

    angle = random.random()*2*np.pi
    cos = np.cos(angle)
    sin = np.sin(angle)
    matrix = np.array([[cos, -sin],
                       [sin, -cos]])

    f = array.shape[0]
    array_xy = np.matmul(array[...,:2].copy().reshape(-1,2),matrix).reshape(f,-1,2)
    array_z = array[...,-1].copy()
    array = np.dstack((array_xy, array_z))

    return array,f

