import numpy as np
import itertools as iter

from tqdm import tqdm

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
        self.data_2d_std = self.data_3d_std[:, :, :, 0:2]
        self.data_3d_preview = self.data_3d_std

    # ***** has_collision() ***** #
    def find_bounding_box_2d(self, person, frame):
        # vertex 1---------- vertex 2
        #    |                  |
        #    |                  |
        # vertex 3---------- vertex 4
        x_max = np.max(self.data_2d_std[person, frame, :, 0])
        x_min = np.min(self.data_2d_std[person, frame, :, 0])
        y_max = np.max(self.data_2d_std[person, frame, :, 0])
        y_min = np.min(self.data_2d_std[person, frame, :, 0])
        bounding_box_2d_vertexbased = np.array([[x_min, y_max], [x_max, y_max], [x_min, y_min], [x_max, y_min]])
        return bounding_box_2d_vertexbased

    def bounding_box_is_overlapped(self, i, another_person, frame):
        bounding_box_2d_i = self.find_bounding_box_2d(i, frame)
        bounding_box_2d_another_person = self.find_bounding_box_2d(another_person, frame)
        # if one vertex of 0 is in the box of 1, return true(overlapped) immediately
        for vertex in range(4):
            if (
                    bounding_box_2d_another_person[0, 0] < bounding_box_2d_i[vertex, 0] <
                    bounding_box_2d_another_person[1, 0] and
                    bounding_box_2d_another_person[2, 1] < bounding_box_2d_i[vertex, 1] <
                    bounding_box_2d_another_person[3, 1]
               ) or (
                    bounding_box_2d_i[0, 0] < bounding_box_2d_another_person[vertex, 0] <
                    bounding_box_2d_i[1, 0] and
                    bounding_box_2d_i[2, 1] < bounding_box_2d_another_person[vertex, 1] <
                    bounding_box_2d_i[3, 1]
               ):
                return True
        return False

    def has_collision(self, i, frame):
        flag = False
        # iterate through all pairs
        for another_person in range(i):
            if self.bounding_box_is_overlapped(i, another_person, frame):
                flag = True
        return flag

    # ***** find_shift_vector() ***** #
    def decide_offset_of_i(self, i, another_person, frame):
        pass

    def find_shift_vector(self):
        shift_vector = np.array([0, 0, 0])
        while True:
            flag_collision = False
            for another_person in range(i):
                if self.bounding_box_is_overlapped_preview(i, another_person, frame):
                    flag_collision = True
                    shift_vector += self.decide_offset_of_i(i, another_person, frame)
                    self.data_3d_preview = self.broadcast_add(self.data_3d_preview, shift_vector)
                    continue
            if flag_collision is False:
                break
        return shift_vector

    # ***** find_max_shift_vector() ***** #
    def find_max_shift_vector(self, shift_vector_list):
        max_shift_vector = np.array([0, 0, 0])
        return max_shift_vector

    # ***** broadcast_add() ***** #
    def broadcast_add(self, i, max_shift_vector):
        result_data_3d_std_for_i = self.data_3d_std[i]
        return result_data_3d_std_for_i

    # ***** main routine ***** #
    def sequential_collision_eliminate_routine(self):
        # the first person do NOT need any kind of shift
        for i in range(1, self.n, 1):
            # the initial value of shift_vector_list is simply DON'T MOVE
            shift_vector_list = np.array([0, 0, 0])

            print("Eliminating person {}'s collision with all other persons".format(i))
            # frame-wise operation, with step size 3 for skipping to accelerate progress
            for frame in tqdm(range(0, self.x, 3)):
                # if the frame of this person has collision with any other person
                if self.has_collision(i, frame):
                    np.append(shift_vector_list, self.find_shift_vector())
                else:
                    np.append(shift_vector_list, np.array([0, 0, 0]))
            max_shift_vector = self.find_max_shift_vector(shift_vector_list)
            self.data_3d_std[i] = self.broadcast_add(i, max_shift_vector)

        return self.data_3d_std
