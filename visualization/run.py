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

    def run():
        """
        The main function 
        """



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
        """
        shape = array_node.shape
        connect = self.info_meta_data["connect"]
        array_vector = torch.cat([Visualization.connect_node(array_node[:,:,:,connect[i][0],:],
                        array_node[:,:,:,connect[i][1],:], i) for i in range(24)], dim = 3)
        array_vector = array_vector.reshape(shape[0],shape[1],shape[2],72)

        return array_vector

    def connect_node(vector_1, vector_2, i):
        return (torch.subtract(vector_2,vector_1)).unsqueeze(3)

    def calculate_axis(vector_1, vector_2):
        axis_unnormalized = torch.dot(vector_1,vector_2)
        return axis_unnormalized/torch.norm(axis_unnormalized)

    def Rodrigue(vect_orig, vect_finl):
        
        axis_unnormalized = torch.dot(vect_orig,vect_finl)
        axis = axis_unnormalized/torch.norm(axis_unnormalized)
        angle = torch.arccos(torch.dot(vect_orig, vect_orig))
        
        r_vect = axis * angle; r_mat = (torch.eye(3) * torch.cos(angle) + 
            (1 - torch.cos(angle)) * torch.matmul(axis.unsqueeze(-1),axis) + 
            torch.sin(angle) * torch.tensor([]))

        return r_vect, r_mat

    def v_cat(axis):

        row1 = torch.cat((torch.cat((torch.zeros_like(axis[:,:,:,:,0]),-axis[:,:,:,:,2]),dim=-1),axis[:,:,:,:,1]),dim=-1)
        row2 = torch.cat((torch.cat((axis[:,:,:,:,2],torch.zeros_like(axis[:,:,:,:,0])),dim=-1),-axis[:,:,:,:,0]),dim=-1)
        row3 = torch.cat((torch.cat((torch.zeros_like(axis[:,:,:,:,0]),-axis[:,:,:,:,2]),dim=-1),axis[:,:,:,:,1]),dim=-1)

        return mat


def calculate_rotational_vector(n, init, finl):
    # rotation axis
    n_unit = torch.divide(n, torch.norm(n))



    return

if __name__ == '__main__':
    v = Visualization()
    