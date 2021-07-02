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
        self.need_select_joint = con.getboolean('skeleton', 'need_select_joint')
        self.select_joint = [int(i) for i in con.get('skeleton', 'select_joint').split(',')]


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
            data = data.reshape(data.shape[0], data.shape[1]//3, 3)
            if self.random_translate:
                data = rf.random_translate(data, translate_distance)
            if self.random_rotate:
                data = rf.random_rotate(data)
            self.data_3d_std.append(data)
        self.data_3d_std = torch.stack([i[:self.frame, :, :] for i in self.data_3d_std])
        if self.need_select_joint:
            self.data_3d_std = self.data_3d_std[:,:,self.select_joint,:]

        print("--Input Info: {}".format(data_path_list))

        return ','.join(data_path_list)

    def main(self):
        
        filename = self.data_preprocess(input_num_min=1, input_num_max=1)

        #collision_handling_process = col_eli(self.data_3d_std)
        #self.data_3d_std = collision_handling_process.collision_eliminate()

        camera = Camera(data = self.data_3d_std,frames=self.frame)
        self.data_2d_std = camera.camera_transform_w2c(self.data_3d_std)

        cov = cover(self.data_2d_std)
        self.cover_std = cov.get_cover_joint()

        self.data_2d_std = camera.camera_transform_c2s(self.data_2d_std)        
        self.data_2d_std[:,:,:,2] = self.cover_std

        print(self.data_2d_std)

        # data = np.array(self.data_2d_std)
        # scio.savemat(self.output_path+filename,{"data":data})

        # visualization after camera switch
        # vis = visc(filename)
        # vis.animate()
        

if __name__ == '__main__':
    test = main()
    test.main()
