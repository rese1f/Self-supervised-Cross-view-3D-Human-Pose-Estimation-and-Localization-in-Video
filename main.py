# system library
from configparser import ConfigParser
import scipy.io as scio
import numpy as np
import torch

# self-created library
import random_function as rf
from visualize import visualize as vis
from collision import collision_eliminate as col_eli

class main:
    def __init__(self,
                 input_num_min=2, input_num_max=2,
                 translate_distance=1000,
                 data_path_list=None  # for testing, use non-random dataset
                 ):

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
        self.shift_vector_list = []

        if self.random_data_source:
            data_path_list = rf.random_data_source(self.input_path, input_num_min, input_num_max)
            print("You've loaded {} successfully".format(data_path_list))
            # why no output?

        # data_path_list == None
        self.data = [torch.tensor(scio.loadmat(i)['data']) for i in [self.input_path + j for j in data_path_list]]

        if self.random_data_timeline:
            for i in range(len(self.data)):
                self.data[i] = rf.random_data_timeline(self.data[i])

        self.frame = np.min([len(i) for i in self.data])
        self.bonding_point = [int(i) for i in con.get('skeleton', 'bonding_point').split(',')]
        self.vertex_number = con.getint('skeleton', 'vertex_number')

        self.translate_distance = translate_distance

    def main(self):
        # [n, x, 32, 3] ==> [x_1 (frame number), 32, 3] + [x_2, 32, 3] + ...
        data_cluster = []
        # without transitions and rotations (v1)
        org = []
        for data in self.data:
            org.append(data.reshape(data.shape[0], test.vertex_number, 3))
        for data in self.data:
            data = data.reshape(data.shape[0], test.vertex_number, 3)
            if self.random_translate:
                data = rf.random_translate(data, self.translate_distance)
            if self.random_rotate:
                data = rf.random_rotate(data)
            data_cluster.append(data)

        # concat data together, [2 people, 3038 total frame number, 32, 3]
        data_cluster = torch.stack([i[:self.frame, :, :] for i in data_cluster])
        print(data_cluster.shape)

        # ***** your code starts here ***** #
        col_eli(data_cluster)
        # ***** your code ends here ***** #

        v1 = vis(org, save_name='before.gif')
        v1.animate()
        v2 = vis(data_cluster, save_name='after.gif')
        v2.animate()

        print(data_cluster)
        print(data_cluster.shape)


if __name__ == '__main__':
    test = main(input_num_min=2, input_num_max=4)
    test.main()
