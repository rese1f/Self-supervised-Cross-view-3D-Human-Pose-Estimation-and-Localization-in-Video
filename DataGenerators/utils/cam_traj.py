# by Wenhao Chai

import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generating script')
    parser.add_argument('--h', default=1500, type=int,
                        help="mean height, unit: mm")
    parser.add_argument('--d', default=2000, type=int,
                        help="mean distance, unit: mm")
    parser.add_argument('--v', default=20, type=int,
                        help="mean velocity, unit: mm per frame")
    
    args = parser.parse_args()
    return args

def cam_traj(pose_3d):
    """this function is to generate a reasonable camera trajction

    Args:
        pose_3d[array](n,f,3)
    
    Returns:
        camera trajction[array](f,3)
    """
    args = parse_args()
    print(args)
    pose_2d = pose_3d.transpose(1,0,2)[...,:2]
    o, r = bounding_circle(pose_2d)
    # init cam
    
def bounding_circle(pose_2d):
    """this function is to generate a reasonable camera trajction

    Args:
        pose_2d (array)[f,n,2]
    
    Returns:
        o (array)[f,[center]]
        r (array){f,[radius]]
    """
    x = pose_2d[...,0]
    y = pose_2d[...,1]
    xmin = np.min(x,axis=1)
    xmax = np.max(x,axis=1)
    ymin = np.min(y,axis=1)
    ymax = np.max(y,axis=1)
    o = np.vstack(((xmin+xmax)/2,(ymin+ymax)/2)).transpose(1,0)
    r = np.sqrt((xmax-xmin)**2+(ymax-xmin)**2)
    return o, r

def random(x:float)-> float:
    """random number with mean x

    Args:
        x (float)

    Returns:
        y (float)
    """
    y = np.random.normal(loc=x, scale=x/10)
    return y

if __name__=='__main__':
    # 2-people 3-frames
    pose_3d = np.array([
        [[1,0,0],
         [1,1,1],
         [2,0,1],],
        [[2,0,2],
         [3,0,1],
         [2,1,1],],
    ])
    cam_traj(pose_3d)
    