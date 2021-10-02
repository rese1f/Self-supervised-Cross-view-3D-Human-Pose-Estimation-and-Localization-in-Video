import torch
import numpy as np
import random


def randomly_load_data():
    """
    Randomly load one person in one frame into the system.
    Returns: the data loaded
    """

    # set scope variables
    input_path = '../VideoPose3D/output/data_output_h36m_1.npz'
    print("\033[1;32mCurrent Input Path\033[0m: " + input_path)

    # get the random sample of one person
    item = np.load(input_path, allow_pickle=True)["positions_3d"].item()
    sampled_item = item[random.sample(range(128), 1)[0]]['pose_pred']
    size = sampled_item.shape
    sampled_item = sampled_item[0, random.sample(range(size[1]), 1)[0], random.sample(range(size[2]), 1)[0]]

    return sampled_item


def subject_to_vector(h36m_metadata, subject):
    """
    The module is used to transfer node tensor into directional vector tensor
    Input: tensor([v,n,x,N,3])
    Output: tensor([v,n,x,24,3])
    """

    connect = h36m_metadata["connect"]
    array_vector = torch.cat([connect_node(subject[connect[i][0], :],
                                           subject[connect[i][1], :])
                              for i in range(24)], dim=-2)
    return array_vector


def connect_node(vector_1, vector_2):
    """
    Assistant function for "subject_to_vector"
    Used for reshaping the tensors
    """
    return (torch.subtract(vector_2, vector_1)).unsqueeze(0)


def apply_to_pose(star_model, h36m_medadata, vector):
    standard_shape = []
    for i in range(3):
        standard_shape.append(torch.tensor(star_model["standard"], dtype=torch.float32))
    connect = h36m_medadata["tree_connect"]
    rotation_vector = torch.zeros_like(vector)
    for i in range(5):
        for j in range(3):
            rotation_vector[connect[i][j], :], r_matrix = Rodrigue(vector[connect[i][j], :],
                                                                   standard_shape[j][connect[i][j], :])

            first_term = r_matrix.unsqueeze(0).repeat(24, 1, 1)\
                .to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            second_term = standard_shape[j].unsqueeze(2)\
                .to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            standard_shape[j] = torch.matmul(first_term, second_term).squeeze()

    return rotation_vector


def Rodrigue(vect_orig, vect_finl):
    """
    The mathematic model for rotation vector and matrix calculation
    Input: []
    """

    # change to the same device
    vect_orig = vect_orig.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    vect_finl = vect_finl.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # shift the axises
    axis_unnormalized = torch.cross(vect_orig, vect_finl, dim=-1)  # 求旋转轴
    axis = torch.divide(axis_unnormalized, torch.norm(axis_unnormalized, dim=-1).unsqueeze(-1))  # 标准化旋转轴
    angle = torch.arccos(torch.dot(vect_orig, vect_orig).unsqueeze(-1))  # 求旋转角度
    r_vect = torch.multiply(axis, angle)  # 求得旋转向量

    # change to the same device
    angle = angle.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    r_vect = r_vect.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Rodrigues 公式，可以求得矩阵R
    eye = torch.eye(3).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    cos = torch.cos(angle).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    first_term = torch.mul(eye, cos)

    second_term = torch.subtract(torch.tensor(1), torch.cos(angle))
    third_term = torch.matmul(axis.unsqueeze(-1).permute(1, 0), axis)
    fourth_term = torch.mul(torch.sin(angle), v_cat(axis))
    r_mat = first_term + torch.mul(second_term, third_term) + fourth_term

    return r_vect, r_mat


def v_cat(axis):
    """
    This module is used for concatcate a matrix used in Rodrigue()
    Input: tensor([v,n,x,N,3])
    Output: tensor([v,n,x,N,3,3])
    """

    axis = axis.unsqueeze(0)
    row1 = torch.cat((torch.cat((torch.zeros_like(axis[..., 0]), -axis[..., 2]), dim=-1), axis[..., 1]),
                     dim=-1).unsqueeze(0)
    row2 = torch.cat((torch.cat((axis[..., 2], torch.zeros_like(axis[..., 0])), dim=-1), -axis[..., 0]),
                     dim=-1).unsqueeze(0)
    row3 = torch.cat((torch.cat((-axis[..., 1], axis[..., 2]), dim=-1), torch.zeros_like(axis[..., 0])),
                     dim=-1).unsqueeze(0)
    mat = torch.cat((torch.cat((row1, row2), dim=-2), row3), dim=-2)

    return mat


def form_star_model(betas, batch_size, poses, star):
    betas = torch.cuda.FloatTensor(betas)
    trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))
    model = star.forward(poses, betas, trans).cpu().detach().numpy()
    return model


def save_model(model, star, obj_index):
    output_path = 'objects/run' + str(obj_index) + '.obj'
    print("\033[1;32mCurrent Output Path\033[0m: " + output_path)
    with open(output_path, 'w') as file:
        for i in model:
            for v in i:
                file.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in star.f + 1:
            file.write('f %d %d %d\n' % (f[0], f[1], f[2]))
