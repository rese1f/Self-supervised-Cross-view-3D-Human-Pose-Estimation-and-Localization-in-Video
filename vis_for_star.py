from star.pytorch.star import STAR
star = STAR(gender='female')
import torch
import numpy as np 
from torch.autograd import Variable
batch_size=1

# 姿态参数theta：24x3=72

poses = np.array([[0., 90., 0.],
                       [0, 0., 0.],
                       [0., 0, 0.],
                       [0., 0, 0.],
                       [0., 0., 0],
                       [0., 0., 0.],
                       [90, 0., 0.],
                       [0., 0, 0.],
                       [0., 0, 0.],
                       [0., 0., 0],
                       [0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0, 0.],
                       [0., 0, 0.],
                       [0., 0, 0.],
                       [0., 0, 0.],
                       [0., 0, 0.],
                       [0., 0, 0.],
                       [0., 0, 0.],
                       [0., 0, 0.],
                       [0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0, 0.],
                       [0., 0., 0.]]).flatten().reshape(1,72)
poses = torch.cuda.FloatTensor(poses)
#poses = torch.cuda.FloatTensor(np.zeros((batch_size,72)))
poses = Variable(poses,requires_grad=True)
# 体型参数beta：10
betas = np.array([
            np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
                      2.20098416, 0.26102114, -3.07428093, 0.55708514,
                      -3.94442258, -2.88552087])])
betas = torch.cuda.FloatTensor(np.zeros((batch_size,10)))
betas = Variable(betas,requires_grad=True)
# 相机参数trans：3
trans = torch.cuda.FloatTensor(np.ones((batch_size,3)))
trans = Variable(trans,requires_grad=True)
d = star(poses, betas,trans)

# 生成STAR的obj文件
d_np = d.cpu().detach().numpy()

outmesh_path = './hello_star.obj'
with open(outmesh_path, 'w') as fp:
    for i in d_np:
        for v in i:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
    for f in star.f + 1:
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
