from os import supports_bytes_environ
import torch
import numpy as np 
from torch.autograd import Variable
from star.pytorch.star import STAR


class cover():
    def __init__(self, data_2d_std):
        '''
        data_2d_std: [n,x,32,3] in (x,y,d)
        '''
        #device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        #self.data = data_2d_std.to(device=device)
        self.data = data_2d_std 
        self.get_cross(self.data)


    def convert(self, data):
        '''
        convert the Human3.6M to the STAR in single frame and single person
        data: [32,3]
        return: [24,3]
        '''
        return torch.stack([
            data[0,:],
            0.8*data[6,:]+0.2*data[7,:],
            0.8*data[1,:]+0.2*data[2,:],
            0.5*data[0,:]+0.5*data[12,:],
            data[4,:],
            data[2,:],
            0.9*data[12,:]+0.1*data[0,:],
            data[8,:],
            data[3,:],
            data[12,:],
            data[10,:],
            data[5,:],
            data[13,:],
            0.5*data[12,:]+0.5*data[17,:],
            0.5*data[12,:]+0.5*data[25,:],
            data[14,:],
            data[17,:],
            data[25,:],
            data[18,:],
            data[26,:],
            data[20,:],
            data[27,:],
            data[22,:],
            data[31,:]
        ])


    def loc2pose(self):
        '''
        convert the joints 3D location to pose info
        '''
        pass


    def get_cross(self, data_2d_std):
        '''
        get the bonding in x-axis and then judge for the cross area
        '''
        x_info = data_2d_std[:,:,:,0]
        
        # bound -> [x,n,2]
        bound = torch.stack([torch.min(x_info,-1).values,torch.max(x_info,-1).values],-1).transpose(0,1).numpy()

        # depth -> [x,n]
        depth = torch.min(data_2d_std[:,:,:,2],-1).values.transpose(0,1).numpy()

        # cross area
        # for loop for single frame
        
        for x in bound:
            # sort for all the point
            v = np.sort(x.flatten())
            # initial the left bound list, id set and cross result list
            left = v[0]
            set, cross = list(), list()
            for right in v:
                index = (x==right).nonzero()
                id = index[0]
                if (index[1] == 0):
                    set.append(id)
                    continue
                else:
                    set.remove(id)
                    cross.append([left,right,set])
                    left = right
            print(cross)
            print('\n')
        
            