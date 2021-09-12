# by Wenhao Chai

import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generating script')
    parser.add_argument('--h', default=1500, type=int,
                        help="mean height, unit: mm")
    parser.add_argument('--d', default=2000, type=int,
                        help="mean distance, unit: mm")
    parser.add_argument('--v', default=10, type=int,
                        help="mean velocity, unit: mm per frame")
    parser.add_argument('--alpha', default=0.001, type=float,
                        help="close coefficient")
    parser.add_argument('--beta', default=0.9, type=float,
                        help="momentum coefficient")
    parser.add_argument('--gamma', default=0.05, type=float,
                        help="shake coefficient")
    
    args = parser.parse_args()
    return args

def cam_traj(pose_3d):
    """this function is to generate a reasonable camera trajction

    Args:
        pose_3d[array]: (n,f,3)
    
    Returns:
        camera trajction[array]: (f,3)
    """
    args = parse_args()
    print(args)
    pose_2d = pose_3d.transpose(1,0,2)[...,:2]
    f = pose_2d.shape[0]
    o, r = bounding_circle(pose_2d)
    # init cam position and velocity
    p = np.empty((f,2))
    v = np.empty((f,2))
    angle = np.random.rand()*2*np.pi
    clockwise = np.random.choice([-1,1])
    d = random(args.d)
    p[0] = o[0] + (r[0] + d)*np.array([np.sin(angle), np.cos(angle)])
    v[0] = np.zeros((2))
    for t in range(1,f):
        dis = np.linalg.norm(p[t-1]-o[t-1])-r[t-1]
        v[t] = (1-args.beta)*(args.alpha*dis*radial_component(p,o,t-1) + random(args.v)*clockwise*tangential_component(p,o,t-1)) + args.beta*v[t-1]
        p[t] = p[t-1] + v[t]
    p_h = vertical_component(args.h,f,args.gamma)
    cam_traj = np.hstack((p, p_h))
    # with constant h
    # cam_traj = np.insert(p,2,values=args.h,axis=1)
    return cam_traj
    
def bounding_circle(pose_2d):
    """this function is to generate a reasonable camera trajction

    Args:
        pose_2d (array): [f,n,2]
    
    Returns:
        o (array): [f,[center]]
        r (array): [f,[radius]]
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

def radial_component(p,o,t):
    """this function returns the radial unit component at time t

    Args:
        p ([array]): [2]
        o ([array]): [2]
        t ([int]): time
    
    Returns:
        v_r ([array]): [2]
    """
    
    v_r = p[t] - o[t]
    v_r /= np.linalg.norm(v_r)
    return v_r

def tangential_component(p,o,t):
    """this function returns the tangential unit component at time t

    Args:
        p ([array]): [2]
        o ([array]): [2]
        t ([int]): time
    
    Returns:
        v_t ([array]): [2]
    """
    rot90 = np.array([[ 0,1],
                      [-1,0]])
    v_t = np.matmul(p[t] - o[t], rot90)
    v_t /= np.linalg.norm(v_t)
    return v_t

def vertical_component(h, f, gamma):
    """this function returns the vertical position of camera

    Args:
        h ([int]): mean and init height
    
    Returns:
        p_h ([array]): [f]
    """
    A = h*gamma
    p_h = A * np.sin(np.linspace(0,f,f)/25*np.pi) + h
    p_h = np.expand_dims(p_h, axis=1)
    return p_h

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
    pose_3d = np.zeros((3,1000,3))
    p = cam_traj(pose_3d)
    import matplotlib.pyplot as plt
    plt.plot(p[:,0],p[:,1])
    plt.scatter([0], [0], s=25, c='r')
    plt.xlim(-3000,3000)
    plt.ylim(-3000,3000)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-3000,3000)
    ax.set_ylim(-3000,3000)
    ax.set_zlim(0,2000)
    ax.plot(p[:,0],p[:,1],p[:,2])
    ax.scatter(0, 0, 0, c='r',marker='o')
    plt.show()
    