import torch
import numpy as np
import itertools as iter  # for permutation
from configparser import ConfigParser

from visualize import visualize as vis
from visualize_2d import visualize_2d as vis_2d

"""
    NOTE(Jack BAI). We utilize PEP-257 comment & programming standard here. Please 
        follow this standard if you want to make changes or add things to this script.
"""


def test_bounding_box(data_cluster_2d, data_cluster):
    test_path_list = ['../Dataexpand_dataset/S1_Directions.mat', '../Dataexpand_dataset/S1_Greeting.mat']
    # try animating the boxes inside 2 dimensions
    v_2d = vis_2d(data=data_cluster_2d, data_path_list=test_path_list, save_name='2d.gif')
    v_2d.animate()
    # compare with 3d case
    v_3d = vis(data=data_cluster, data_path_list=test_path_list, save_name='3d.gif')
    v_3d.animate()
    if True:
        print("test_bounding_box: SUCCESS")
    if False:
        print("test_bounding_box: FAIL")
    return


def throw_vertex_part(data_cluster_2d):
    con = ConfigParser()
    con.read('configs.ini')

    # frame is a number indicating the shorter one for the two videos
    x = np.min([data_cluster_2d[i].shape[0]
                for i in range(len(data_cluster_2d))])
    # bounding points are decided: [0, 12, 21, 18, 26, 29, 16, 3, 5, 8, 10]
    bounding_point = [int(i) for i in con.get('skeleton', 'bonding_point').split(',')]

    # initialize the tensor cube
    tensor_cube = []

    # iterate through n
    for data in data_cluster_2d:
        # data_cluster_2d has torch.Size([n, x, 32, 2])
        # data has torch.Size([x, 32, 2]), with n iteration rounds

        # initialize the tensor plane
        tensor_plane = []

        # iterate through x
        for frame in range(x):
            # candidate vertices for the box
            x_sus = [data[frame, i, 0] for i in bounding_point]
            y_sus = [data[frame, i, 1] for i in bounding_point]

            # switch into the tensor
            # x_sus is a list, and x_max is a value (the maximum of the list)
            x_sus = torch.stack(x_sus, dim=0)
            y_sus = torch.stack(y_sus, dim=0)
            x_max = torch.max(x_sus)
            x_min = torch.min(x_sus)
            y_max = torch.max(y_sus)
            y_min = torch.min(y_sus)

            # make a tensor bar
            x_max_tensor = torch.tensor([x_max])
            y_max_tensor = torch.tensor([y_max])
            x_min_tensor = torch.tensor([x_min])
            y_min_tensor = torch.tensor([y_min])
            tensor_bar = torch.stack((x_max_tensor, y_max_tensor,
                                      x_min_tensor, y_min_tensor))

            # make a tensor plane, concat if possible
            if tensor_plane == []:
                tensor_plane = tensor_bar
            else:
                tensor_plane = torch.cat((tensor_bar, tensor_plane), dim=1)

        # make a tensor cube, concat if possible
        # case 1. do not exist
        if tensor_cube == []:
            tensor_cube = tensor_plane  # becomes [4, x]
            flag = 1
        # case 2. only added once
        elif flag == 1:
            tensor_cube = torch.stack((tensor_cube, tensor_plane), dim=0)  # becomes [2, 4, x]
            flag = 0
        # case 3. added more than once
        else:
            tensor_plane = torch.unsqueeze(tensor_plane, 0)
            tensor_cube = torch.cat((tensor_cube, tensor_plane), dim=0)  # becomes [3, 4, x]

    bounding_box_cluster = tensor_cube.permute(0, 2, 1)
    return bounding_box_cluster


def find_2d_bounding_box(data_cluster):
    """Utility function for finding the 2-dimensional bounding box

    Find the 2-dimensional bounding box (with the top-down view) of one person.
    In details, deal with [x, 32, 2] for each i in n, by slicing dimensions into 2
    matrices. Iterate through all people and slice dimensions into 2. The initial
    output should be [x, 2], but because we have 2 dimensions we gain [x, 4]. By
    iterating through n people we get [n, x, 4].

    Args:
        data_cluster: The tensor of size [n, x, 32, 3], where
            n: number of people;
            x: number of frames;
            32: number of vertices on a person;
            3: number of dimensions for each vertex.

    Returns:
        The bounding boxes of each person, as a tensor of size [n, x, 8] where
            n: number of people;
            x: number of frames;
            4: x_max, x_min, y_max, y_min (2 vertices, 2 dimensions, 2*2 = 4)

    Raises:
        NOError: no error occurred up to now
    """

    # slice 3-dimensional to 2-dimensional
    indices = torch.tensor([0, 1])
    data_cluster_2d = torch.index_select(data_cluster, 3, indices)
    # print(data_cluster_2d.shape) gives torch.Size([2, x, 32, 2]), because z-axis is
    #   sliced out.

    # test if the 2D bounding box is successfully found
    # test_bounding_box(data_cluster_2d, data_cluster)

    # throw the vertex part
    bounding_box_cluster = throw_vertex_part(data_cluster_2d)

    # return the bunch of bounding boxes
    return bounding_box_cluster


def find_central_distance(bounding_box):
    """Utility function for finding the distance of two bounding boxes.

    For each pair of bounding boxes, find the distance between them, using their
    central points. Read distances of EACH FRAME (each j in x). For each pair
    of people there will be [x], and for n people there will be [n, n, x],
    as [n, n] is the upper-triangular adjacency matrix of the people.

    Args:
        bounding_box: The tensor of size [n, x, 4].

    Returns:
        The distance tensor of size [n, n, x] for each frame

    Raises:
        NOError: no error occurred up to now
    """

    # initialize the adjacency matrix tensor
    n = bounding_box.shape[0]
    x = bounding_box.shape[1]
    adjacency_matrix_tensor = torch.zeros(n, n, x)

    # find the central point tensor of bounding_box
    tensor_x = bounding_box[:, :, 0:2]
    tensor_y = bounding_box[:, :, 2:4]

    central_tensor_x = torch.mean(input=tensor_x, dim=2, keepdim=True)  # [n, x, 1 for x_center]
    central_tensor_y = torch.mean(input=tensor_y, dim=2, keepdim=True)  # [n, x, 1 for y_center]
    central_tensor_xy = torch.cat((central_tensor_x, central_tensor_y), dim=2)  # [n, x, 2 for (x,y)_center]

    # transform the central point tensor to the permutation of central distances [n, n, x]

    # find all the possibilities and store them into one list
    permutation_list = list(iter.combinations([i for i in range(n)], 2))  # [(0, 1), (0, 2), (1, 2)]

    # the matrix form should be
    # [[ 0, 1, 1
    #    0, 0, 1
    #    0, 0, 0 ]]

    # iterate through the list and reshape the distances into the distance tensor
    for permutation in permutation_list:
        coordinate_of_0_x = central_tensor_xy[permutation[0], :, 0]
        coordinate_of_0_y = central_tensor_xy[permutation[0], :, 1]
        coordinate_of_1_x = central_tensor_xy[permutation[1], :, 0]
        coordinate_of_1_y = central_tensor_xy[permutation[1], :, 1]

        distance_of_01_x = coordinate_of_0_x - coordinate_of_1_x  # [x]
        distance_of_01_y = coordinate_of_0_y - coordinate_of_1_y

        distance_of_01 = (distance_of_01_x.pow(2) + distance_of_01_y.pow(2)).pow(0.5)  # [x]
        adjacency_matrix_tensor[permutation] = distance_of_01

    distance_tensor = adjacency_matrix_tensor  # [n, n, x]
    return distance_tensor


def transfer_line_to_vertex_box(bounding_box):
    # 1----------2
    # |          |
    # |          |
    # 3----------4

    # localization
    x_max_tensor = bounding_box[:, :, 0]
    y_max_tensor = bounding_box[:, :, 1]
    x_min_tensor = bounding_box[:, :, 2]
    y_min_tensor = bounding_box[:, :, 3]

    # new order: x_min, y_max, x_max, y_max, x_min, y_min, x_max, y_min
    bounding_box_vertexbased = torch.stack((
        x_min_tensor, y_max_tensor, x_max_tensor, y_max_tensor,
        x_min_tensor, y_min_tensor, x_max_tensor, y_min_tensor
    ), dim=2)

    return bounding_box_vertexbased


def find_collision_cases(bounding_box, bounding_box_vertexbased):
    """Utility function for finding the

    After finding the distances between multiple boxes, it's time to find the
    candidate collision cases. We utilize the "candidate method", which is also
    utilized in visualize.py, to gain a result tensor of size [n, n, x].

    Args:
        bounding_box: The tensor of size [n, x, 4].

    Returns:
        All the collision cases, as a list of tensor of size [n, n, x]. Inside the tensor all values
        must be 0 (for no collision) and 1 (for collision)

    Raises:
        NOError: no error occurred up to now
    """

    # initialize the adjacency matrix tensor
    n = bounding_box.shape[0]
    x = bounding_box.shape[1]
    adjacency_matrix_tensor = torch.zeros(n, n, x)

    # find all the possibilities and store them into one list
    permutation_list = list(iter.combinations([i for i in range(n)], 2))  # [(0, 1), (0, 2), (1, 2)]

    # iterate through the list of [0, 1], [0, 2], [1, 2] and find the collision cases
    for permutation in permutation_list:

        # find the box range of both items
        x_range_max_0 = bounding_box[permutation[0], :, 0]
        y_range_max_0 = bounding_box[permutation[0], :, 1]
        x_range_min_0 = bounding_box[permutation[0], :, 2]
        y_range_min_0 = bounding_box[permutation[0], :, 3]

        x_range_max_1 = bounding_box[permutation[1], :, 0]
        y_range_max_1 = bounding_box[permutation[1], :, 1]
        x_range_min_1 = bounding_box[permutation[1], :, 2]
        y_range_min_1 = bounding_box[permutation[1], :, 3]

        # decide whether any vertex of one term is in the range of another term

        # any one vertex of four suffices
        for i in range(4):
            vertex_coordination_x_0 = bounding_box_vertexbased[permutation[0], :, i]
            vertex_coordination_y_0 = bounding_box_vertexbased[permutation[0], :, i + 1]

            vertex_coordination_x_1 = bounding_box_vertexbased[permutation[1], :, i]
            vertex_coordination_y_1 = bounding_box_vertexbased[permutation[1], :, i + 1]

            # iterate through each fream
            for frame in range(bounding_box.shape[1]):
                # if the vertex is in the range
                if (
                        (x_range_max_0[frame] > vertex_coordination_x_1[frame] > x_range_min_0[frame] and
                         y_range_max_0[frame] > vertex_coordination_y_1[frame] > y_range_min_0[frame]) or
                        (x_range_max_1[frame] > vertex_coordination_x_0[frame] > x_range_min_1[frame] and
                         y_range_max_1[frame] > vertex_coordination_y_0[frame] > y_range_min_1[frame])
                ):
                    # mark the frame
                    adjacency_matrix_tensor[permutation][frame] = 1

    collision_cases_list = adjacency_matrix_tensor
    return collision_cases_list


def decide_shift_vector(collision_cases_list):
    """Utility function for finding the perfect shift vector for all frames

    Firstly, we construct the shift vectors of all the collision cases, which
    contains 2 possibilities for each frame, as we want to gain a better realization
    of different direction of shifts. Secondly, we utilize a mathematical method on
    the list of all shift vectors to choose the vector with largest absolute value
    of length. This leads to a perfect match by conjecture.

    Args:
        collision_cases_list: the list containing collision case vectors, each of
            size [n_1, n_2, x_0].

    Returns:
        The answer of the best shift_vector.

    Raises:
        NOError: no error occurred up to now
    """
    best_shift_vector = []
    return best_shift_vector


def collision_eliminate(data_cluster):
    """Wrapper function for collision elimination

    Reconstruct content in the cluster of data, from [2, x, 32, 3] to [2, x_star, 32, 3]

    Args:
        data_cluster: The tensor of size [2, numOfFrames(or we call "x"), 32, 3].

    Returns:
        Literally returns nothing, but changes data_cluster in the memory heap directly.

    Raises:
        NOError: no error occurred up to now
    """

    bounding_box = find_2d_bounding_box(data_cluster)  # [n, x, 4]
    print("the shape of bounding_box is {}".format(bounding_box.shape))
    central_distance_tensor = find_central_distance(bounding_box)
    print("the shape of distance tensor is {}".format(central_distance_tensor.shape))
    bounding_box_vertexbased = transfer_line_to_vertex_box(bounding_box)
    print("the shape of vertex-based bounding box is {}".format(bounding_box_vertexbased.shape))
    collision_cases_list = find_collision_cases(bounding_box, bounding_box_vertexbased)
    print("the shape of collision cases list is {}".format(collision_cases_list.shape))
    best_shift_vector = decide_shift_vector(collision_cases_list)
    # data_cluster += best_shift_vector
    return  # nothing
