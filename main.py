from configparser import ConfigParser
import scipy.io as scio
import numpy as np
import random_function as rf
import torch
from visualize import visualize as vis
from collision import collision_eliminate as col_eli


class main:
    def __init__(self,
                 input_num_min=2, input_num_max=4,
                 translate_distance=1000,
                 data_path_list=["S1_Discussion.mat", "S1_Greeting.mat", "S1_Purchases 1.mat"]):

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

        if self.random_data_source:
            data_path_list = rf.random_data_source(self.input_path, input_num_min, input_num_max)

        print("You've load {} successfully".format(data_path_list))

        self.data = [torch.tensor(scio.loadmat(i)['data']) for i in [self.input_path + j for j in data_path_list]]

        if self.random_data_timeline:
            for i in range(len(self.data)):
                self.data[i] = rf.random_data_timeline(self.data[i])

        self.frame = np.min([len(i) for i in self.data])
        self.bonding_point = [int(i) for i in con.get('skeleton', 'bonding_point').split(',')]
        self.vertex_number = con.getint('skeleton', 'vertex_number')

        self.translate_distance = translate_distance

    def camera(self, data_cluster, in_camera_data_cluster, ex_camera_data_cluster):
        """
        n: 人数;
        x: 帧数;
        data_cluster: [n,x,32,3];
        in_camera_data_cluster: 固定的内参矩阵 [3,3];
        ex_camera_data_cluster: 随帧数变化的外参矩阵 [x,3,4];
        return: [n,x,32,2];
        """
        pass

    def data_preprocess(self):
        self.data_3d_std = []
        for data in self.data:
            data = data.reshape(data.shape[0], test.vertex_number, 3)
            if self.random_translate:
                data = rf.random_translate(data, self.translate_distance)
            if self.random_rotate:
                data = rf.random_rotate(data)
            self.data_3d_std.append(data)
        self.data_3d_std = torch.stack([i[:self.frame, :, :] for i in self.data_3d_std])

    def main(self):
        
        self.data_preprocess()

        collision_handling_process = col_eli(self.data_3d_std)
        self.data_3d_std = collision_handling_process.collision_eliminate()

        visualize_process = vis(self.data_3d_std, save_name='after.gif')
        visualize_process.animate()


if __name__ == '__main__':
    test = main()
    test.main()
