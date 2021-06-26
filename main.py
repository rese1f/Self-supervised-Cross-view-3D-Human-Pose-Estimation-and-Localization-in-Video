from configparser import ConfigParser
import scipy.io as scio
import numpy as np
import random_function as rf
import torch
from visualize import visualize as vis
from collision import collision_eliminate as col_eli
from camera import Camera
from visualize_camera import visualize_2d as visc
from cover import cover

class main:
    def __init__(self):

        if torch.cuda.is_available():
            torch_device = torch.device('cuda')

        con = ConfigParser()
        con.read('configs.ini')

        self.input_path = con.get('path', 'input_path')
        self.output_path = con.get('path', 'output_path')
        self.random_data_source = con.getboolean('random_optional', 'random_data_source')
        self.random_data_timeline = con.getboolean('random_optional', 'random_data_timeline')
        self.random_translate = con.getboolean('random_optional', 'random_translate')
        self.random_rotate = con.getboolean('random_optional', 'random_rotate')
        self.bonding_point = [int(i) for i in con.get('skeleton', 'bonding_point').split(',')]
        self.vertex_number = con.getint('skeleton', 'vertex_number')
        self.head_index = con.getint('skeleton', 'head')
        self.body_index = [int(i) for i in con.get('skeleton', 'body').split(',')]
        self.leg_index = [int(i) for i in con.get('skeleton', 'leg').split(',')]
        self.arm_index = [int(i) for i in con.get('skeleton', 'arm').split(',')]
        

    def data_preprocess(self, data_path_list=None, input_num_min=2, input_num_max=4, translate_distance=1000):
        if self.random_data_source:
            data_path_list = rf.random_data_source(self.input_path, input_num_min, input_num_max)
        self.data = [torch.tensor(scio.loadmat(i)['data']) for i in [self.input_path + j for j in data_path_list]]

        if self.random_data_timeline:
            for i in range(len(self.data)):
                self.data[i] = rf.random_data_timeline(self.data[i])

        self.frame = np.min([len(i) for i in self.data])
        
        self.data_3d_std = []
        for data in self.data:
            data = data.reshape(data.shape[0], test.vertex_number, 3)
            if self.random_translate:
                data = rf.random_translate(data, translate_distance)
            if self.random_rotate:
                data = rf.random_rotate(data)
            self.data_3d_std.append(data)
        self.data_3d_std = torch.stack([i[:self.frame, :, :] for i in self.data_3d_std])
        print("--Input Info: {}".format(data_path_list))

        return ','.join(data_path_list)

    def main(self):
        
        filename = self.data_preprocess(input_num_min=3)

        #collision_handling_process = col_eli(self.data_3d_std)
        #self.data_3d_std = collision_handling_process.collision_eliminate()


        camera_1 = Camera(self.frame)
        self.data_2d_std = camera_1.camera_transform_w2c(self.data_3d_std)

        cov = cover(self.data_2d_std)
        
        return

        self.data_2d_std = camera_1.camera_transform_c2s(self.data_2d_std)        
        self.data_2d_std[:,:,:,2] = self.cover_std

        data = np.array(self.data_2d_std)
        scio.savemat(self.output_path+filename,{"data":data})

        vis = visc(filename)
        vis.animate()
        

        

if __name__ == '__main__':

    test = main()
    test.main()
