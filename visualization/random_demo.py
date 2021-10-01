from star.star import STAR
import numpy as np
from util.data_utils import *
from util.helper import *
import torch

# define the basic global variables
betas = np.array([
    np.array([2.25176191, -3.7883464, 0.46747496, 3.89178988,
              2.20098416, 0.26102114, -3.07428093, 0.55708514,
              -3.94442258, -2.88552087])])
num_betas = 10
batch_size = 1
obj_number = 5
star = STAR(gender='female', num_betas=num_betas)

for i in range(obj_number):
    # load data into the runtime system
    subject = randomly_load_data()  # unit test passed
    # transform the tensors of our subjects into the corresponding vectors for the STAR model
    vector = subject_to_vector(h36m_metadata, subject)  # unit test passed
    # apply the vectors on the STAR pose
    rotation_vector = apply_to_pose(star_model, h36m_metadata, vector)
    poses = rotation_vector
    # form the STAR model
    model = form_star_model(betas, batch_size, poses, star)
    # save the file
    save_model(model, star, i)
