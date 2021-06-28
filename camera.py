import numpy as np
import torch
from configparser import ConfigParser

torch.set_default_tensor_type(torch.DoubleTensor)


class Camera:

    def __init__(self, data=None, frames=10, tracking=1) -> None:
        """Initialization & Localization

        Initialization part with introduction to all the local variables.

        Args:
            self: the object entity.
            data: the data to use camera on.
            frames: the number of frames.
            tracking: flag to decide whether the camera aims on the center point or not.
                true: yes, the camera will always aiming at the center point.
                false: no, the camera will always keep the same direction.

        Returns:
            The ctor of a class will return nothing.

        Raises:
            TypeError: do NOT change "if data is None" to "if data == None", because "is" is often used in "if"
                statement.
        """
        self.ifTracking = tracking
        self.frame = frames
        # x_default = np.linspace(-200,200,frames,dtype = np.float16).reshape(frames,1);

        self.dir_initial = np.array([-np.pi / 2, -np.pi, 0])
        # no data input
        if data is None:
            self.pos_initial = np.array([0., -3500., 1500.])
            self.center = np.array([0., 0., 0.])

        # have data input, no tracking
        elif tracking == 1:
            Camera.__get_center(self, data=data)
            self.pos_initial = self.center + [0., -3500., 1500.]

        # have data input, have tracking
        elif tracking == 0:
            Camera.__get_center(self, data=data)
            self.pos_initial = self.center + [0., -3500., 1500.]
            self.center = np.array([0., 0., 0.])

        x_default = np.array([self.pos_initial[0]] * frames, dtype=np.float16).reshape(frames, 1)
        y_default = np.array([self.pos_initial[1]] * frames, dtype=np.float16).reshape(frames, 1)
        z_default = np.array([self.pos_initial[2]] * frames, dtype=np.float16).reshape(frames, 1)
        self.camera_pos = np.concatenate((np.concatenate((x_default, y_default), 1), z_default), 1)

        dir_x_default = np.array([self.dir_initial[0]] * frames, dtype=np.float16).reshape(frames, 1)
        dir_y_default = np.array([self.dir_initial[1]] * frames, dtype=np.float16).reshape(frames, 1)
        dir_z_default = np.array([self.dir_initial[2]] * frames, dtype=np.float16).reshape(frames, 1)

        self.camera_arg = np.concatenate((np.concatenate((dir_x_default, dir_y_default), 1), dir_z_default), 1)

        con = ConfigParser()
        con.read('configs.ini')
        self.fx = con.getfloat('camera_parameter', 'fx')
        self.fy = con.getfloat('camera_parameter', 'fy')
        self.u = con.getfloat('camera_parameter', 'u')
        self.v = con.getfloat('camera_parameter', 'v')
        self.s = con.getfloat('camera_parameter', 's')

        '''
        Generate extrinsic and intrinsic camera matrix from its default parameter
        These matrix could be updated later
        '''
        Camera.__exmat_generator(self)
        Camera.__inmat_generator(self, self.fx, self.fy, self.u, self.v, self.s)

        return

    def __exmat_generator(self):
        """Generate the extrinsic matrix for the camera.

        Develop the extrinsic matrix for the camera, which is used for transforming the object in the world coordinate
        system into the homogeneous camera coordinate system. Note that the intermediate variable H_o2k (from outside
        coordinate system to the camera coordinate system) has the size
        [ R T    where R is the rotational matrix, calculated by the camera orientation, with R_z * R_y * R_x.
          0 1 ]  T is the translational matrix, calculated by -R*(-1) * X, where X is the camera position in the
                    world coordinate.

        Args:
            self.frame: the number of frames in the animation.
            self.camera_pos: the position of the camera.
            self.camera_arg: 1

        Returns:
            self.exmat: of size [x, 4, 4], where x is the number of frames (self.frames), so it's frame-dependent.

        Raises:
            NOError: no error occurred up to now
        """

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

            H_o2k = torch.cat((R, T), 1)
            H_o2k = torch.cat((H_o2k, torch.tensor([[0., 0., 0., 1.]])), 0)

            if i == 0:
                self.exmat = H_o2k
            elif i == 1:
                self.exmat = torch.stack((self.exmat, H_o2k))
            else:
                self.exmat = torch.cat((self.exmat, H_o2k.reshape(1, 4, 4)), 0)

        return

    def __inmat_generator(self, fx=1527.4, fy=1529.2, u=957.1, v=529.8, s=0):
        """Generate the intrinsic matrix for the camera.

        Develop the intrinsic matrix for the camera, which is used for transforming the object in the camera coordinate
        system into the homogeneous pixel coordinate system. Note that the K matrix has size
        [ f_x, s,   x_0 (u) ]
        [ 0,   f_y, y_0 (v) ]
        [ 0,   0,   1       ]
        For more details you can check the private blog concerning this project at http://jackgetup.com/.

        Args:
            self:
            fx: the x-axis focal distance
            fy: the y-axis focal distance
            u: coordinate tilt in the x direction
            v: coordinate tilt in the y direction
            s: the corresponds of the principle point

        Returns:
            self.inmat: of size [3, 3], not frame-dependent

        Raises:
            NOError: no error occurred up to now
        """

        # TODO: The distortions need to be added.

        self.inmat = torch.tensor(np.array([[fx, s, u], [0, fy, v], [0, 0, 1]]))
        # self.inmat[0,0] = fx
        # self.inmat[2,2] = fy
        # self.inmat[0,1] = u
        # self.inmat[2,1] = v

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
            datasT[i] = torch.matmul(self.exmat[i], datasT[i]);
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

    def __get_center(self, data):
        slice1 = data[:, :, 15, 0]
        slice2 = data[:, :, 15, 1]

        sum1 = torch.sum(slice1)
        sum2 = torch.sum(slice2)

        self.center = np.array([sum1, sum2, 0]) / 3
        pass

    def get_angle(frame,data):
        if frame == None:
            result = torch.zeros(3)
            for i in range(3):
                x = data[i]; y = data[(i+1)%3]; 
                result[(i+2)%3] = torch.atan2(y,x);
        else:
            result = torch.zeros((frame,3))
            for i in range(3):
                x = data[:,i]; y = data[:,(i+1)%3]; 
                result[:,(i+2)%3] = torch.atan2(y,x);
        return result



    def camMotion_linear_motion(self, velocity = 10, dir = np.array([1,0,0]), tracking = 0):
        """
        velocity: the speed of the camera, unit mm/frame
        dir: the directional vector of the velocity, should be a unit vector
        tracking: the parameter determine whether the camera will point at the center defined in __init__
        the camera doing constant speed motion with a fixed direction
        """
        dir = dir/np.linalg.norm(dir)
        frames = self.frame
        self.camera_pos = torch.add(self.camera_pos,torch.mul(torch.tensor(dir*velocity),torch.tensor(np.linspace(0,frames,frames)).reshape(frames,1)));
        if tracking != 0:
            dirVec = torch.sub(self.center,self.camera_pos)
            self.camera_arg = self.get_angle(frames,self.camera_pos)
        
        return


    def camMotion_rotation(self, data, radius=3000, angle=np.pi * 2):
        '''
        radius: the radius of rotation
        angle: the entire arc the carmera travels
        get 2d rotation around center with given radius and angle (the center is defined in __init__ )
        '''
        
        frames = self.frame

        x_2center = np.cos(np.linspace(0,angle,frames,dtype = np.float16).reshape(frames,1)) * radius;
        y_2center = np.sin(np.linspace(0,angle,frames,dtype = np.float16).reshape(frames,1)) * radius;
        z_2center = np.array([0] * frames, dtype=np.float16).reshape(frames, 1)
        xyz_2center = np.concatenate((np.concatenate((x_2center, y_2center), 1), z_2center), 1)

        self.camera_pos = torch.add(torch.tensor([self.center]*frames).reshape(frames,3), torch.tensor(xyz_2center));
        self.camera_arg = self.get_angle(frames, torch.sub(self.center,self.camera_pos))

        return



    def update_camera(self):
        '''
        Updating the extrinsics matrix
        '''
        Camera.__exmat_generator(self)

        pass


