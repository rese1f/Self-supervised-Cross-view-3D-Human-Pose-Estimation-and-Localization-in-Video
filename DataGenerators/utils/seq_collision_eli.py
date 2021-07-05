import numpy as np
import itertools as iter
import tqdm

"""
    Note(Jack BAI). This script the approved script for collision elimination, and the old version
        (collision.py) is used to COMPARE with this one.
"""


class sequential_collision_elimination:
    def __init__(self, data_3d_std=None):
        # local variable initialization
        self.n = data_3d_std.shape[0]
        self.x = data_3d_std.shape[1]
        self.number_of_vertices = data_3d_std.shape[2]
        self.data_3d_std = data_3d_std

        # intermediate variable initialization


        # result variable initialization
        self.result_data_3d_std = self.data_3d_std

    def has_collision(self):
        pass

    def find_shift_vector(self):
        pass

    def find_max_shift_vector(self, shift_vector_list):
        max_shift_vector = np.array([0, 0, 0])
        return max_shift_vector

    def broadcast_add(self, i, max_shift_vector):
        result_data_3d_std_for_i = self.data_3d_std[i]
        return result_data_3d_std_for_i

    def sequential_collision_eliminate_routine(self):
        for i in range(self.n):
            # the initial value of shift_vector_list is simply DON'T MOVE
            shift_vector_list = np.array([0, 0, 0])
            for frame in range(0, self.x, 3):
                if self.has_collision():
                    np.append(shift_vector_list, self.find_shift_vector())
                else:
                    np.append(shift_vector_list, np.array([0, 0, 0]))
            max_shift_vector = self.find_max_shift_vector(shift_vector_list)
            self.result_data_3d_std[i] = self.broadcast_add(i, max_shift_vector)

        print(np.array([1, 2, 3]))

        return self.result_data_3d_std
