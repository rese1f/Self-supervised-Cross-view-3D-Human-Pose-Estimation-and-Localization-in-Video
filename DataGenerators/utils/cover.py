from os import supports_bytes_environ
import torch
import numpy as np 
from torch.autograd import Variable
from star.pytorch.star import STAR


class cover():
    def __init__(self, data_2d_std):
        '''
        data_2d_std: [n,x,17,3] in (x,y,d)
        '''
        self.data = data_2d_std
        self.record = torch.zeros_like(self.data[:,:,:,0])


    def get_cross_cases(self):
        '''
        get the bonding in x-axis and then judge for the cross area
        m * [frame, [xmin, xmax], list(id)]
        list[id] sorted in depth
        '''
        x_info = self.data[:,:,:,0]
        
        # bound -> [x,n,2] + depth -> [x,n]
        bound = torch.stack([torch.min(x_info,-1).values,torch.max(x_info,-1).values,torch.min(self.data[:,:,:,2],-1).values],-1).transpose(0,1).numpy()

        # cross area
        # for loop for single frame

        self.cross_cases = list()
        for i in range(len(bound)):
            x = bound[i]
            # sort for all the point
            v = np.sort(x[:,:2].flatten())
            # initial the left bound list, id set and cross result list
            left = v[0]
            set = list()
            # traversal for the sorted list as right bound
            for right in v:
                index = (x==right).nonzero()
                id = int(index[0])
                d = x[id,2]            
                if (index[1] == 0):
                    set.append(d)
                else:
                    if (len(set) > 1):
                        set.sort()      
                        self.cross_cases.append([i,[left,right],[int((j==x).nonzero()[0]) for j in set]])
                    set.remove(d)
                left = right
        
        return
    

    def generate_cover(self, frame, id):
        '''
        generate the cover area due to the id and frame
        beta version: use bonding box
        plan: use STAR model
        '''
        return torch.min(self.data[id,frame,:,1]),torch.max(self.data[id,frame,:,1])


    def get_cover_joint(self):
        self.get_cross_cases()
        # case -> [frame, [xmin, xmax], list(id)]
        for case in self.cross_cases:
            frame = case[0]
            xmin = case[1][0]
            xmax = case[1][1]
            # simplify the least depth to be the cover
            cov = self.generate_cover(frame,case[2][0])
            # for beta version, get the cover
            ymin = cov[0]
            ymax = cov[1]
            # consider the covered joint
            for id in case[2][1:]:
                joint_x = self.data[id,frame,:,0]
                joint_y = self.data[id,frame,:,1]
                for i in range(self.data.shape[2]):
                    if (xmin < joint_x[i] < xmax) and (ymin < joint_y[i] < ymax):
                        self.record[id,frame,i] = 1

        return self.record
