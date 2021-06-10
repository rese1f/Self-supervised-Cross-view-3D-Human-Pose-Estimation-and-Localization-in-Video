import torch
import numpy as np

# note that we utilize PEP-257 comment & programming standard here.


def find_2d_bonding_box(data_cluster):
    """Effective function for finding the 2-dimensional bonding box



    """
    pass


def collision_eliminate(data_cluster):
    """Wrapper function for collision elimination

    Reconstruct content in the cluster of data, from [2, x, 32, 3] to [2, x_star, 32, 3]

    Args:
        data_cluster: The torch of size [2, numOfFrames(or we call "x"), 32, 3].

    Returns:
        Literally returns nothing, but changes data_cluster in the memory heap directly.

    Raises:
        NOError: no error occurred up to now
    """
    # TODO(Jack): finish the utilized part of this function
    return # nothing
