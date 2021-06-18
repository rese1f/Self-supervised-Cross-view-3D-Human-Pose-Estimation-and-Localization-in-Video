import numpy as np
import torch
from configparser import ConfigParser

torch.set_default_tensor_type(torch.DoubleTensor)


class Camera:

    def __init__(self, frames) -> None:
        '''
        Constructing an object of Camera class, defining its frame number, default parameters
        '''

        self.frame = frames
        # x_default = np.linspace(-200,200,frames,dtype = np.float16).reshape(frames,1);

        self.pos_initial = np.array([0., -3500., 1500.])

        x_default = np.array([self.pos_initial[0]] * frames, dtype=np.float16).reshape(frames, 1)
        y_default = np.array([self.pos_initial[1]] * frames, dtype=np.float16).reshape(frames, 1)
        z_default = np.array([self.pos_initial[2]] * frames, dtype=np.float16).reshape(frames, 1)
        self.camera_pos = np.concatenate((np.concatenate((x_default, y_default), 1), z_default), 1)

        self.dir_initial = np.array([-np.pi / 2, -np.pi, 0])

        dir_x_default = np.array([self.dir_initial[0]] * frames, dtype=np.float16).reshape(frames, 1)
        dir_y_default = np.array([self.dir_initial[1]] * frames, dtype=np.float16).reshape(frames, 1)
        dir_z_default = np.array([self.dir_initial[2]] * frames, dtype=np.float16).reshape(frames, 1)

        self.camera_arg = np.concatenate((np.concatenate((dir_x_default, dir_y_default), 1), dir_z_default), 1)

        con = ConfigParser();
        con.read('configs.ini');
        self.fx = con.getfloat('camera_parameter', 'fx')
        self.fy = con.getfloat('camera_parameter', 'fy')
        self.u = con.getfloat('camera_parameter', 'u')
        self.v = con.getfloat('camera_parameter', 'v')
        self.s = con.getfloat('camera_parameter', 's')

        '''
        Gernerate extrinsics and intrinsics camera matrix from its default parameter
        These matrix could be updated later
        '''
        Camera.__exmat_generator(self)
        Camera.__inmat_generator(self, self.fx, self.fy, self.u, self.v, self.s)

        pass

    def __exmat_generator(self):
        '''
        Generate extrinsics matrix, used for transform the object in homogeneous world coordinate into homogeneous camera coordinate

        H_o2k = [ R  T ]
                [ 0  1 ]  used for homogeneous coordinate

        R is rotational matrix calculated from camera orination, Rz*Ry*Rx
        T is -R^-1*X ; X here is camera position in world coordinate
        '''
        for i in range(self.frame):
            posCamera = self.camera_pos[i]
            argCamera = self.camera_arg[i]

            Rz = torch.tensor(np.array(
                [[np.cos(argCamera[2]), -np.sin(argCamera[2]), 0.], [np.sin(argCamera[2]), np.cos(argCamera[2]), 0.],
                 [0., 0., 1.]], dtype=np.float16))
            Ry = torch.tensor(np.array([[np.cos(argCamera[1]), 0., np.sin(argCamera[1])], [0., 1., 0.],
                                        [-np.sin(argCamera[1]), 0., np.cos(argCamera[1])]], dtype=np.float16))
            Rx = torch.tensor(np.array([[1., 0., 0.], [0., np.cos(argCamera[0]), -np.sin(argCamera[0])],
                                        [0., np.sin(argCamera[0]), np.cos(argCamera[0])]], dtype=np.float16))

            R = torch.mm(Rz, Ry)
            R = torch.mm(R, Rx)

            T = - torch.mm(R, torch.tensor(np.array(posCamera, dtype=np.float16).reshape(3, 1)))

            H_ok = torch.cat((R, T), 1)
            H_ok = torch.cat((H_ok, torch.tensor([[0., 0., 0., 1.]])), 0)

            if i == 0:
                self.exmat = H_ok
            elif i == 1:
                self.exmat = torch.stack((self.exmat, H_ok))
            else:
                self.exmat = torch.cat((self.exmat, H_ok.reshape(1, 4, 4)), 0)

        return

    def __inmat_generator(self, fx=1527.4, fy=1529.2, u=957.1, v=529.8, s=0):
        '''
        Generate intrinsics matrix
        * NOT COMPLETED
        '''

        self.inmat = torch.tensor(np.array([[fx, s, u], [0, fy, v], [0, 0, 1]]))
        # self.inmat[0,0] = fx;
        # self.inmat[2,2] = fy;
        # self.inmat[0,1] = u;
        # self.inmat[2,1] = v;
        # self.inmat

        return

    def camera_transform_w2c(self, data_3d):
        """
        n: 人数;
        x: 帧数;
        data_cluster: [n,x,32,3];
        in_camera_data_cluster: 固定的内参矩阵 [3,3];
        ex_camera_data_cluster: 随帧数变化的外参矩阵 [x,3,4];
        return: [n,x,32,2];

        Transform the tensor in homogeneous 3d world coordinate into euclidiean
        equivlant ot 3d camera coordinate (homogeneous 2d coordinate)

        """

        dataShape = data_3d.size()
        x = dataShape[0]
        n = dataShape[1]

        datasInHC = torch.cat((data_3d, torch.tensor(np.ones((x, n, 32, 1), dtype=np.float16))), dim=3).reshape(x, n, 32, 4, 1)


        datasT = torch.transpose(datasInHC,0,1);
        # 交换person与frame维度

        for i in range(self.frame):
            datasT[i] = torch.matmul(self.exmat[i],datasT[i]);
            i += 1;
        
        #datasT = torch.matmul(self.exmat,datasT);
        #3维的广播方法不可用，即[frame,4,4]*[frame,32,4,1]
        #正在尝试[frame,32,4,4]*[frame,32,4,1]

        # datasT = torch.matmul(self.inmat,datasT);
        # 旧版本操作，新版本中该操作移动至camera_transform_c2s()

        datasT = torch.transpose(datasT, 0, 1).reshape(x, n, 32, 4)[:, :, :, :3]
        #datasT[:,:,:,1] = - datasT[:,:,:,1]

        return datasT

    def camera_transform_c2s(self, data):
        '''
        Transform the tensor into sensor coordinate and eliminate the depth demention
        '''
        data = data.reshape([data.shape[0], data.shape[1], 32, 3, 1])
        data = torch.matmul(self.inmat, data);
        data = data / torch.abs(torch.unsqueeze(data[:, :, :, 2], 3))
        data = data.reshape([data.shape[0], data.shape[1], 32, 3])

        return data

    def get_rotation_center(self, data):

        self.rotational_center = np.array([torch.sum(data[:, :, 15, 0]), torch.sum(data[:, :, 15, 1])]) / 3

        return

    def camMotion_rotation(self, data, center=None, radius=3000, angle=np.pi * 2):
        return

    def update_camera(self):
        '''
        Updating the rigid-motion parameters and intrinsics parameters
        * NOT COMPLETED
        '''

        pass


