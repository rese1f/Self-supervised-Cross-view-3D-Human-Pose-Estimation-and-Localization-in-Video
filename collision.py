import torch
import numpy as np

"""
    NOTE(Jack BAI). We utilize PEP-257 comment & programming standard here. Please 
        follow this standard if you want to make changes or add things to this script.
"""


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
        The bounding boxes of each person, as a tensor of size [n, x, 4] where
            n: number of people;
            x: number of frames;
            4: x_1, x_2, y_1, y_2 (2 vertices, 2 dimensions, 2*2 = 4)

    Raises:
        NOError: no error occurred up to now
    """

    bounding_box = []
    # TODO(Jack): finish the utilized part of this function

    return bounding_box


def find_distance(bounding_box):
    """Utility function for finding the distance of two bounding boxes.

    For each pair of bounding boxes, find the distance between them, using their
    central points. Read distances of EACH FRAME (each j in x). For each pair
    of people there will be [x, 1], and for n people there will be [n, n, x, 1],
    as [n, n] is the upper-triangular adjacency matrix of the people.

    Args:
        bounding_box: The tensor of size [n, x, 4].

    Returns:
        The distance tensor of size [n, n, x, 1] for each frame

    Raises:
        NOError: no error occurred up to now
    """

    distance_tensor = []
    return distance_tensor


def find_collision_cases(distance_tensor):
    """Utility function for finding the

    After finding the distances between multiple boxes, it's time to find the
    candidate collision cases. We utilize the "candidate method", which is also
    utilized in visualize.py, to gain a result tensor of size [n_1, n_2, x_0].

    Args:
        distance tensor: The distance tensor of size [n, n, x, 1] for each frame

    Returns:
        All the candidate collision cases, as as list of tensor of size
        [n_1, n_2, x_0] where
            n_1: person 1 who collides with person 2
            n_2: person 2 who collides with person 1
            x_0: the particular frame number where the pair of people collides

    Raises:
        NOError: no error occurred up to now
    """
    collision_cases_list = []
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

    bounding_box = find_2d_bounding_box(data_cluster)
    distance_tensor = find_distance(bounding_box)
    collision_cases_list = find_collision_cases(distance_tensor)
    best_shift_vector = decide_shift_vector(collision_cases_list)
    # data_cluster += best_shift_vector
    return  # nothing
