from numpy.core.numeric import zeros_like
from star.star import STAR
import numpy as np
from numpy import newaxis
import pickle
import os
import errno
import sys
import torch
from tqdm import tqdm
from util.data_utils import *
from util.arguments import parse_args


"""
input

output_zip = {
        sample'0': {
            'pose_pred': list(v,n,x,17,3),
            'T': list(v,n,x,3),
            'ground': [a,b,c],
            'receptive_field': int
        }
        sample'1': {
            'pose_pred': list(v,n,x,17,3),
            'T': list(v,n,x,3),
            'ground': [a,b,c],
            'receptive_field': int
        }
    }

"""


class Visualization():
    """
    Structure of the data in class

    data:
        data_prediction: dict(
            int(sample): pose-> ndarray [v,n,x,17,3]
        )

    info:
        info_args: argument
        info_samplelist: 




    
    """


    def __init__(self) -> None:

        self.info_args = parse_args()
        print(self.info_args)

        Visualization.load_data(self)
        Visualization.create_path(self)
        Visualization.load_model(self)

        print(self.data_prediction[1].shape)
        self.data_vector = Visualization.node_2_vector(self, self.data_prediction[1])
        Visualization.run(self)

        pass

    def load_data(self):
        try:
            if self.info_args.inputpath is not None:
                filepath = self.info_args.inputpath

            else:
                filepath = 'data/input/data_output_' + self.info_args.dataset + '_1.npz'
            #filepath = 'output/data_output_' + self.info_args.dataset + '_1.npz'
            self.info_filepath = filepath

            print("\n")
            print("Current File: " + filepath)
            print("\n")

            prediction_orig = np.load(filepath, allow_pickle=True)["positions_3d"].item()
            self.samplelist = prediction_orig.keys(); self.data_prediction = dict()            
            for i in self.samplelist: self.data_prediction[i] = Visualization.comb(prediction_orig[i]) 
            
        except FileNotFoundError:
            print('File does not exist! Please input right file path')

        self.info_meta_data = suggest_metadata(self.info_args.dataset)
        self.info_std_shape = suggest_metadata("star_model")
        self.data_std_shape = []
        for i in range(3): self.data_std_shape.append(torch.tensor(self.info_std_shape["standard"]))

        return

    def create_path(self):
        self.info_shape = dict()
        for sample in self.samplelist:
            self.info_shape[sample] = self.data_prediction[sample].shape
            shape = self.info_shape[sample]
            for i in range(shape[0]):
                try:
                    for j in range(shape[1]):
                        os.mkdir("data\output\\" + str(i) + "_" + "j")
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise RuntimeError('Unable to create output directory')
            
        return

    def load_model(self):
        """
        Load the STAR model
        """

        self.data_betas = np.array([
                    np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
                            2.20098416, 0.26102114, -3.07428093, 0.55708514,
                            -3.94442258, -2.88552087])])
        self.info_num_betas=10
        self.info_batch_size=1
        self.data_star = STAR(gender='female',num_betas=self.info_num_betas)
        return

    def visualize():


        return

    def save_data(self, model, outmesh_path):

        #outmesh_path = './hello_star.obj'
        with open(outmesh_path, 'w') as fp:
            for i in model:
                for v in i:
                    fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
            for f in self.data_star.f+1:
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
        print('Done.')

        return

    def run(self):
        """
        The main function 
        """
        for i in range(3): 
            shape = self.data_vector.shape
            self.data_std_shape[i].unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(shape[0],shape[1],shape[2],1,1)

        print(self.data_std_shape[1].shape)
        

        self.data_rotvect = torch.zeros_like(self.data_vector)
        order_list = self.info_meta_data["tree_connect"]
        for i in range(5):
            Visualization.iterator(self, order_list[i])


        return

    # tool function below

    def comb(prediction):

        if torch.is_tensor(prediction["pose_pred"]):
            pos_pred = prediction["pose_pred"].permute(3,0,1,2,4); pos_trans = prediction["T"]
        else:
            pos_pred = torch.tensor((prediction["pose_pred"]).transpose(3,0,1,2,4)); pos_trans = torch.tensor(prediction["T"])

        return (torch.add(pos_pred, pos_trans[:,:,:])).permute(1,2,3,0,4)

    def node_2_vector(self,array_node):
        """
        The module is used to transfor node tensor into directional vector tensor
        Input: tensor([v,n,x,N,3])
        Output: tensor([v,n,x,24,3])
        """
        shape = array_node.shape
        connect = self.info_meta_data["connect"]
        array_vector = torch.cat([Visualization.connect_node(array_node[:,:,:,connect[i][0],:],
                        array_node[:,:,:,connect[i][1],:], i) for i in range(24)], dim = -2)
        #array_vector = array_vector.reshape(shape[0],shape[1],shape[2],72)
        
        #shape = array_node.shape
        #skeleton = self.info_meta_data["skeleton"]

        #array_vector = torch.cat([Visualization.connect_node(array_node[:,:,:,skeleton[i][0],:],
        #                array_node[:,:,:,skeleton[i][1],:], i) for i in range(len(skeleton))], dim = -2)

        return array_vector

    def connect_node(vector_1, vector_2, i):
        """
        Assistant function for "node_2_vector"
        """
        return (torch.subtract(vector_2,vector_1)).unsqueeze(3)

    def calculate_axis(vector_1, vector_2):
        axis_unnormalized = torch.dot(vector_1,vector_2)
        return axis_unnormalized/torch.norm(axis_unnormalized)

    def Rodrigue(vect_orig, vect_finl):
        """
        The mathematic model for rotation vector and matrix calculation
        Input: []
        """

        
        axis_unnormalized = torch.cross(vect_orig, vect_finl, dim = -1) # 求旋转轴
        axis = torch.divide(axis_unnormalized,torch.norm(axis_unnormalized,dim=-1).unsqueeze(-1)) # 标准化旋转轴
        angle = torch.arccos(torch.dot(vect_orig, vect_orig, dim = -1).unsqueeze(-1)) # 求旋转角度
        
        r_vect = torch.multiply(axis, angle) # 求得旋转向量
        
        # Rodrigues 公式，可以求得矩阵R
        r_mat = (torch.matmul(torch.eye(3), torch.cos(angle)) + 
            torch.mul((torch.subtract(torch.tensor(1),torch.cos(angle)),torch.matmul(axis.unsqueeze(-1).permute(0,1,2,3,5,4),axis))) + 
            torch.mul(torch.sin(angle),Visualization.v_cat(axis)))

        return r_vect, r_mat

    def v_cat(axis):
        """
        This module is used for concatcate a matrix used in Rodrigue()
        Input: tensor([v,n,x,N,3])
        Output: tensor([v,n,x,N,3,3])
        """

        row1 = torch.cat((torch.cat((torch.zeros_like(axis[...,0]),-axis[...,2]),dim=-1),axis[...,1]),dim=-1).unsqueeze(4)
        row2 = torch.cat((torch.cat((axis[...,2],torch.zeros_like(axis[...,0])),dim=-1),-axis[...,0]),dim=-1).unsqueeze(4)
        row3 = torch.cat((torch.cat((torch.zeros_like(axis[...,0]),-axis[...,2]),dim=-1),axis[...,1]),dim=-1).unsqueeze(4)
        mat = torch.cat((torch.cat((row1,row2),dim=-1),row3),dim=-1)
        
        return mat

    def iterator(self, pair):
        '''
        This module is used for iteration to get rotational vector
        '''

        for i in range(3):
            self.data_rotvect[:,:,:,i,:], R_mat = Visualization.Rodrigue(self.data_std_shape[i][:,:,:,i,:], self.data_vector[:,:,:,i,:])
            self.data_std_shape[i][:,:,:,:,:] = torch.matmul(R_mat, self.data_std_shape[i][:,:,:,:,:].unsqueeze(5).permute(3,0,1,2,5,4)).permute(1,2,3,0,5,4).unsqueeze(5)
        


        return


def calculate_rotational_vector(n, init, finl):
    # rotation axis
    n_unit = torch.divide(n, torch.norm(n))



    return

if __name__ == '__main__':
    v = Visualization()
    