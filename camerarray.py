import numpy as np
from configparser import ConfigParser
from camera import Camera




class CameraArrary:
    def __init__(self, data, frames, tracking = 1, type_list = "phone") -> None:

        self.camera_list_type = type_list.split(",")
        self.camera_array = []
        
        for cam in self.camera_list_type:
            obj = Camera(data=data, frames=self.frame, tracking=tracking, type=cam)
            self.camera_array.append(obj)
        
        
        pass

    def do_transform_w2c(self, data_wc):

        data_cc = []
        for camera in self.camera_array:
            data_cc.append(camera.camera_transform_w2c(data_wc))
        data_cc = np.array(data_cc)

        return data_cc

    def do_transform_c2s(self, data_cc):

        data_sc = []
        for i in range(len(self.camera_list_type)):
            data_sc.append(self.camera_array[i].camera_transform_c2s(data_cc[i]))
        data_sc = np.array(data_sc)

        return data_sc

    def get_extrinsics_parameter(self):

        ex_list = []
        for i in range(len(self.camera_list_type)):
            ex_list.append([self.camera_list_type[i], self.camera_array[i].camera_pos, self.camera_array[i].camera_arg])


        return ex_list

    def save_camera_array():
        pass



