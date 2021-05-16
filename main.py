from configparser import ConfigParser
import scipy.io as scio
import numpy as np
import random_function as rf
import torch
from visualize import visualize as vis


class main():
    def __init__(self, 
            input_num_min = 2, input_num_max = 2, 
            translate_distance = 1000, 
            data_path_list = None): 

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
            data_path_list  = rf.random_data_source(self.input_path, input_num_min, input_num_max)
            print("You've load {} successfully".format(data_path_list))

        self.data = [torch.tensor(scio.loadmat(i)['data']) for i in [self.input_path + j for j in data_path_list]]

        if self.random_data_timeline:
            for i in range(len(self.data)):
                self.data[i] = rf.random_data_timeline(self.data[i])

        self.frame = np.min([len(i) for i in self.data])
        self.bonding_point = [int(i) for i in con.get('skeleton','bonding_point').split(',')]
        self.vertex_number = con.getint('skeleton','vertex_number')

        self.translate_distance = translate_distance


    def if_collision(self, datas):
        '''
        check if any two people collision
        '''
        pass


    def main(self):
        datas = []
        org = []
        for data in self.data:
            org.append(data.reshape(data.shape[0],test.vertex_number,3))
        for data in self.data:
            data = data.reshape(data.shape[0],test.vertex_number,3)
            if self.random_translate:
                data = rf.random_translate(data,self.translate_distance)
            if self.random_rotate:
                data = rf.random_rotate(data)
            datas.append(data)
        datas = torch.stack([i[:self.frame,:,:] for i in datas])
        #self.if_collision(datas)
        
        
        v1 = vis(org,save_name='before.gif')
        v1.animate()
        v2 = vis(datas,save_name='after.gif')
        v2.animate()


        print(datas)
        print(datas.shape)

if __name__ == '__main__':
    test = main(input_num_min = 2, input_num_max = 4)
    test.main()