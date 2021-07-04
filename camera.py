import numpy as np
#import torch
from configparser import ConfigParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#torch.set_default_tensor_type(torch.DoubleTensor)
'''
[Alert] this project have changed all the tensor of Pytorch into ndarray of Numpy!! Be sure to input proper datatype!
'''

class Camera:

    def __init__(self, data=None, frames=10, tracking=1, type = "phone", loadpath = None) -> None:
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

            

        if loadpath is not None:
            Camera.__load_camera(self, loadpath)
            if type == "phone":
                Camera.phone(self,load = 1)
            elif type == "monitor":
                pass
        else:

            self.ifTracking = tracking
            self.frame = frames

            dataShape = data.size()
            self.x = dataShape[0]
            self.n = dataShape[1]
            self.y = dataShape[2]

            try:
                Camera.__get_center(self,data)
            except:
                print("data needed")
            pass

            if type == "phone":
                Camera.phone(self,data)
            elif type == "monitor":
                pass
        
        #Camera.__visc(self)

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
            self.camera_arg: the tri-rotation arguments of the camera (incl. x, y, z axes)

        Returns:
            self.exmat: of size [x, 4, 4], where x is the number of frames (self.frames), so it's frame-dependent.

        Raises:
            NOError: no error occurred up to now
        """

        for i in range(self.frame):
            posCamera = self.camera_pos[i]
            argCamera = self.camera_arg[i]

            Rz = np.array(
                [[np.cos(argCamera[2]), -np.sin(argCamera[2]), 0.], [np.sin(argCamera[2]), np.cos(argCamera[2]), 0.],
                 [0., 0., 1.]], dtype=np.float16)
            Ry = np.array([[np.cos(argCamera[1]), 0., np.sin(argCamera[1])], [0., 1., 0.],
                                        [-np.sin(argCamera[1]), 0., np.cos(argCamera[1])]], dtype=np.float16)
            Rx = np.array([[1., 0., 0.], [0., np.cos(argCamera[0]), -np.sin(argCamera[0])],
                                        [0., np.sin(argCamera[0]), np.cos(argCamera[0])]], dtype=np.float16)

            R = np.matmul(Rz, Ry)
            R = np.matmul(R, Rx)
            R = R.transpose(0,1)
            self.R = R

            T = - np.matmul(R, np.array(posCamera, dtype=np.float16).reshape(3, 1))

            H_o2k = np.concatenate((R, T), 1)
            H_o2k = np.concatenate((H_o2k, np.array([[0., 0., 0., 1.]])), 0)

            if i == 0:
                self.exmat = H_o2k.reshape(1, 4, 4)
            else:
                self.exmat = np.concatenate((self.exmat, H_o2k.reshape(1, 4, 4)), 0)

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

        self.inmat = np.array([[fx, s, u], [0, fy, v], [0, 0, 1]])

        return

    def camera_transform_w2c(self, data_3d):
        """Use the extrinsic matrix to switch the graph from world coordinate to camera coordinate

        Transform the tensor in homogeneous 3d world coordinate into euclidean
        equivalent to 3d camera coordinate (homogeneous 2d coordinate).

        Args:
            data_3d: the 3-D coordinates of the raw data, i.e. data_cluster, of size [n, x, 17, 3]

        Variables:
            n: the number of people
            x: the number of frames

        Returns:
            datasT: of size [n, x, 17, 3]

        Raises:
            NOError: no error occurred up to now
        """

        dataShape = data_3d.size()
        x = dataShape[0]
        n = dataShape[1]
        y = dataShape[2]

        datasInHC = np.concatenate((data_3d, np.ones((x, n, y, 1), dtype=np.float16)), dim=3)\
            .reshape(x, n, y, 4, 1)

        # 交换person与frame维度
        datasT = np.transpose(datasInHC, 0, 1);

        for i in range(self.frame):
            datasT[i] = np.matmul(self.exmat[i], datasT[i])
            i += 1

        datasT = np.transpose(datasT, 0, 1).reshape(x, n, y, 4)[:, :, :, :3]
        #datasT[:, :, :, 2] =  - datasT[:, :, :, 2]

        return datasT

    def camera_transform_c2s(self, data):
        """Use the intrinsic matrix to switch the graph from camera coordinate to pixel coordinate

        Transform the tensor in homogeneous camera coordinate into euclidean
        equivalent to pixel coordinate (2d to 2d).

        Args:
            data: the 3-D coordinates of the raw data, i.e. data_cluster, data_3d, all of size [n, x, 17, 3]

        Returns:
            data: of the same size as input data, i.e. [n, x, 17, 3]

        Raises:
            NOError: no error occurred up to now
        """
        data = data.reshape([data.shape[0], data.shape[1], self.y, 3, 1])
        data = np.matmul(self.inmat, data)
        data = data / np.abs(np.unsqueeze(data[:, :, :, 2], 3))
        data = data.reshape([data.shape[0], data.shape[1], self.y, 3])

        return data

    def __get_center(self, data):
        """Get the center of all the users

        Get the center coordinates of all the humans in the plot. Note that the center point is on the horizontal
        plane, so it has a z-value of 0.

        Args:
            data: the 3-D coordinates of the raw data, i.e. data_cluster, data_3d, all of size [n, x, 17, 3]

        Returns:
            datas: of the same size as input data, i.e. [n, x, 17, 3]

        Raises:
            NOError: no error occurred up to now
        """
        slice1 = data[:, 0, 10, 0]
        slice2 = data[:, 0, 10, 1]
        sum1 = np.sum(slice1)
        sum2 = np.sum(slice2)

        self.center = np.array([sum1, sum2, 0]) / 3
        print(self.center)
        return

    def get_angle(frame, data):
        """
        Get the camera euclidean angle from given directional vector

        Args:
            frame: the total frame of the output array
            data: directional vector data

        Returns:

            [frame,3] ndarray of euclidean angle
        """
        if frame is None:
            result = np.zeros(3)
            x = data[0]
            y = data[1]
            z = data[2]
            l = np.sqrt(np.add(np.mul(x,x), np.mul(y,y)))
            result[2] = np.atan2(y, x)
            result[1] = (np.add(np.atan2(z, l),np.pi/2))
        else:
            result = np.zeros((frame, 3))
            
            x = data[:, 0]; y = data[:, 1]; z = data[:, 2]
            l = np.sqrt(np.add(np.mul(x,x), np.mul(y,y)))
            result[:, 2] = (np.add(np.atan2(y, x),np.pi/2))
            result[:, 0] =  np.pi/2# (torch.sub(torch.tensor(np.pi/2),torch.atan2(z, l)))
            #print(result)
        
        return result

    def cam_motion_linear_motion(self, velocity=10, dir=np.array([1, 0, 0]), tracking=0):
        """
        generate camera position vector of constant speed linear motion with a fixed direction 

        Args:
            self: the obj itself
            velocity: the speed of the camera, unit mm/frame
            dir: the directional vector of the velocity, should be a unit vector
            tracking: the parameter determine whether the camera will point at the center defined in __init__

        Returns:

            Null
        """
        dir = dir/np.linalg.norm(dir)
        frames = self.frame
        self.camera_pos = np.add(self.camera_pos,np.mul(dir*velocity),np.linspace(0,frames,frames).reshape(frames,1));
        if tracking != 0:
            dirVec = np.sub(self.center,self.camera_pos)
            self.camera_arg = self.get_angle(frames,dirVec)
        else:
            dirVec = np.sub(self.center,self.camera_pos[0])
            self.camera_arg[:,:] = Camera.get_angle(dirVec)
        
        return

    def cam_motion_linear_motion2(self, sta_point, end_point, tracking = 0):
        """
        generate camera position vector of constant speed linear motion with known start and end point

        Args:
            self: the obj itself
            sta_point & end_point: ndarray(float16) of start and end point
            tracking: the parameter determine whether the camera will point at the center defined in __init__

        Returns:

            Null
        
        """
        frames = self.frame
        self.camera_pos = np.linspace(sta_point, end_point, frames);
        if tracking != 0:
            dirVec = np.sub(self.center,self.camera_pos)
            self.camera_arg = Camera.get_angle(frames,dirVec)
        else:
            dirVec = np.sub(self.center,self.camera_pos[0])
            self.camera_arg[:,:] = Camera.get_angle(dirVec)
        
        return

    def cam_motion_rotation(self, data, radius=3000, angle=np.pi * 2):
        """
        generate camera position vector of constant speed rotation around center with given radius and angle
        * the center is the geometry center of individuals in dataset

        Args:
            self: the obj itself
            radius: the radius of rotation
            angle: the entire arc the carmera travels

        Returns:

            Null

        """
        frames = self.frame

        x_2center = np.cos(np.linspace(0,angle,frames,dtype = np.float16).reshape(frames,1)) * radius;
        y_2center = np.sin(np.linspace(0,angle,frames,dtype = np.float16).reshape(frames,1)) * radius;
        z_2center = np.array([0] * frames, dtype=np.float16).reshape(frames, 1)
        xyz_2center = np.concatenate((np.concatenate((x_2center, y_2center), 1), z_2center), 1)

        self.camera_pos = np.add([self.center]*frames.reshape(frames,3), xyz_2center);
        self.camera_arg = self.get_angle(frames, self.center,self.camera_pos)

        return

    def __read_parameter(self,type = "phone"):
        '''
        function used for reading intrinsics parameter from files

        Args:
            self: the obj itself
            type: camera type
        
        Returns:

            Null

        Raises:
            more config required for different types of camera, eg. monitor, drones
        '''

        con = ConfigParser();
        con.read('configs.ini');
        self.fx = con.getfloat('camera_parameter', 'fx')
        self.fy = con.getfloat('camera_parameter', 'fy')
        self.u = con.getfloat('camera_parameter', 'u')
        self.v = con.getfloat('camera_parameter', 'v')
        self.s = con.getfloat('camera_parameter', 's')

        return

    def phone(self, data, pointList = None, motionType = "line", load = 0):
        '''
        This func is used for generating a camera looks like phone view

        Args: 
            self: the object itself
            data: original 3d data
            pointList: the specific start and end point, default None
            motionType: what type of motion the camera should do: default constant speed linear motion
            load: used to control whether read already generated camera parameter

        Returns:

            Null

        '''

        if load == 0:
            pass
        else:
            if pointList is None:
                while True:
                    start = np.array(self.center) + np.append(np.random.randint(-10000,10000,size = (2)),np.array([1300]));
                    end = np.array(self.center) + np.append(np.random.randint(-10000,10000,size = (2)),np.array([1300]));
                    
                    if motionType == "line":
                        Camera.cam_motion_linear_motion2(self, sta_point = start, end_point = end, tracking = self.ifTracking);
                    elif motionType == "circle":
                        pass
                    else:
                        pass
                    
                    if Camera.collision_check(data, self.camera_pos):
                        break
            else:
                start = pointList[0];
                end = pointList[1];

                if motionType == "line":
                    Camera.cam_motion_linear_motion2(self, sta_point = start, end_point = end, tracking = self.ifTracking);
                elif motionType == "circle":
                    pass
                else:
                    pass


        Camera.__read_parameter(self,"phone")
        Camera.__exmat_generator(self)
        Camera.__inmat_generator(self, self.fx, self.fy, self.u, self.v, self.s)


        return

    def collision_check(man,pth):
        '''
        the func used to check whether the path of the camera go through one's body
        default man's width is 200 mm

        Args:
            man: n,x,17,3 ndarray represent data of human
            pth: the camera path

        Returns:

            1: if no collision occures
            0: if collisions occure
        '''
        man = man[:,:,10,:].reshape(man.shape[0],man.shape[1],man.shape[3])
        for i in range(man.shape[0]):
            if np.min(np.sqrt(np.sum(np.square(man[i] - pth), axis=1))) <= 200:
                return 0
        return 1

    def update_camera(self):
        """
        Updating the extrinsic matrix

        **this function is have no function until now!!
        """
        Camera.__exmat_generator(self)
        Camera.__inmat_generator(self)

        pass

    def save_camera(self, number):
        '''
        the function used for saving the extrinsics parameter of camera into harddrive, in order for later use

        Args: 
            self: object itself
            number: the number of the camera in camera array

        Rerurns:

            Null

        '''

        save_file = np.array([self.camera_pos,self.camera_arg])
        save_pth = "data\camera_" + str(number) + ".npy"
        np.save(save_pth,save_file)

        pass

    def __load_camera(self, pth):
        '''
        the function used to load extrinsics parameter of a saved camera

        Args:
            self: the obj itself
            pth: saved camera file path

        Returns:

            Null

        '''

        tmp = np.load(pth)
        self.camera_pos = tmp[0]
        self.camera_arg = tmp[1]

        pass

    def __visc(self):
        """
        A function with TONS OF BUGS, plz DO NOT use it
        
        """


        ax = plt.figure().add_subplot(projection='3d')
        dir = np.array(np.matmul(self.R,np.array([0,0,1000]*self.frame,\
            dtype = np.float16).reshape(self.frame,3,1)).reshape(self.frame,3))
        pos = np.array(self.camera_pos)
        x = torch.stack((torch.tensor(pos[:, 0]), torch.tensor(pos[:, 0] + dir[:, 0])), 0)
        y = torch.stack((torch.tensor(pos[:, 1]), torch.tensor(pos[:, 1] + dir[:, 1])), 0)
        z = torch.stack((torch.tensor(pos[:, 2]), torch.tensor(pos[:, 2] + dir[:, 2])), 0)
        
        
        """
        ax.quiver(pos[:,0], pos[:,1], pos[:,2],\
            dir[:,0],\
            dir[:,1],\
            dir[:,2],\
            length=0.1)
        """
        #ax.quiver(pos[:,0], pos[:,1], pos[:,2], dir,length=0.1)
        ax.plot3D(x, y, z, lw=2)

        plt.show()
        return





def dist(vect1,vect2):
    """
    used for reference
    """

    try:
        return np.sqrt(np.sum(np.square(vect1 - vect2), axis=1))

    except:
        print("plz input the same dimension")

