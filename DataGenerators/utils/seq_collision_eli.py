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
        # vertex 0---------- vertex 1
        #    |                  |
        #    |                  |
        # vertex 2---------- vertex 3
        x_max = np.max(self.data_2d_std[person, frame, :, 0])
        x_min = np.min(self.data_2d_std[person, frame, :, 0])
        y_max = np.max(self.data_2d_std[person, frame, :, 1])
        y_min = np.min(self.data_2d_std[person, frame, :, 1])
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
                    bounding_box_2d_another_person[1, 1]
               ) or (
                    bounding_box_2d_i[0, 0] < bounding_box_2d_another_person[vertex, 0] <
                    bounding_box_2d_i[1, 0] and
                    bounding_box_2d_i[2, 1] < bounding_box_2d_another_person[vertex, 1] <
                    bounding_box_2d_i[1, 1]
               ):
                return True
        return False

    def has_collision(self, i, frame):
        flag = False
        # iterate through all pairs
        for another_person in range(i):
            if self.bounding_box_is_overlapped(i, another_person, frame):
                flag = True
                break
        return flag

    # ***** find_shift_vector() ***** #
    def find_bounding_box_2d_preview(self, person, frame):
        # vertex 1---------- vertex 2
        #    |                  |
        #    |                  |
        # vertex 3---------- vertex 4
        x_max = np.max(self.data_3d_preview[person, frame, :, 0])
        x_min = np.min(self.data_3d_preview[person, frame, :, 0])
        y_max = np.max(self.data_3d_preview[person, frame, :, 1])
        y_min = np.min(self.data_3d_preview[person, frame, :, 1])
        bounding_box_2d_vertexbased = np.array([[x_min, y_max], [x_max, y_max], [x_min, y_min], [x_max, y_min]])
        return bounding_box_2d_vertexbased

    def bounding_box_is_overlapped_preview(self, i, another_person, frame):
        bounding_box_2d_i_preview = self.find_bounding_box_2d_preview(i, frame)
        bounding_box_2d_another_person_preview = self.find_bounding_box_2d_preview(another_person, frame)
        # if one vertex of 0 is in the box of 1, return true(overlapped) immediately
        for vertex in range(4):
            if (
                    bounding_box_2d_another_person_preview[0, 0] < bounding_box_2d_i_preview[vertex, 0] <
                    bounding_box_2d_another_person_preview[1, 0] and
                    bounding_box_2d_another_person_preview[2, 1] < bounding_box_2d_i_preview[vertex, 1] <
                    bounding_box_2d_another_person_preview[1, 1]
            ) or (
                    bounding_box_2d_i_preview[0, 0] < bounding_box_2d_another_person_preview[vertex, 0] <
                    bounding_box_2d_i_preview[1, 0] and
                    bounding_box_2d_i_preview[2, 1] < bounding_box_2d_another_person_preview[vertex, 1] <
                    bounding_box_2d_i_preview[1, 1]
            ):
                return True
        return False

    def decide_offset_of_i(self, i, another_person, frame):
        # We rule that the invalid point must be shifted to due north (upper) or due south (down)

        # decide which case it is in
        y_i_max = np.max(self.data_3d_preview[i, frame, :, 1])
        y_i_min = np.min(self.data_3d_preview[i, frame, :, 1])
        y_another_person_max = np.max(self.data_3d_preview[another_person, frame, : ,1])
        y_another_person_min = np.min(self.data_3d_preview[another_person, frame, : ,1])

        # we always shift i, not the other person; create 2 possible offsets in different directions
        if (y_i_max > y_another_person_max > y_i_min > y_another_person_min) or \
                (y_another_person_max > y_i_max > y_another_person_min > y_i_min):
            offset_value_1 = (y_i_min - y_another_person_max) * 1.7  # coefficient is 1.7
            offset_value_2 = (y_i_max - y_another_person_min) * 1.7
        else:
            offset_value_1 = (y_i_max - y_another_person_min) * 1.7
            offset_value_2 = (y_i_min - y_another_person_max) * 1.7

        if np.abs(offset_value_1) > np.abs(offset_value_2):
            offset = np.array([0, offset_value_1, 0])
        else:
            offset = np.array([0, offset_value_2, 0])
        return offset  # e.g. np.array([0, -2, 0])

    def broadcast_add_preview(self, i, frame, offset):
        # : for broadcast; shift_vector should be 3-dimensional
        self.data_3d_preview[i, frame, :] += offset

    def find_shift_vector(self, i, frame):
        shift_vector = np.array([0, 0, 0], dtype=float)
        # start listening
        while True:
            flag_collision = False
            for another_person in range(i):
                # FIXIT: here is a bug in self.bounding_box_is_overlapped_preview()!
                if self.bounding_box_is_overlapped_preview(i, another_person, frame):
                    flag_collision = True
                    offset = self.decide_offset_of_i(i, another_person, frame)
                    shift_vector += offset
                    self.broadcast_add_preview(i, frame, offset)
                    continue
            # if there is collision one collision with any other boxes, do col_eli again
            if flag_collision is False:
                break
        return shift_vector

    # ***** find_max_shift_vector() ***** #
    def find_max_shift_vector(self, shift_vector_list):
        max_shift_vector = np.array([0, 0, 0])
        for shift_vector in shift_vector_list:
            if np.abs(shift_vector[1]) > np.abs(max_shift_vector[1]):
                max_shift_vector = shift_vector
        return max_shift_vector

    # ***** broadcast_add() ***** #
    def broadcast_add_data_3d_std(self, i, max_shift_vector):
        self.data_3d_std[i, :, :] += max_shift_vector

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
                    shift_vector_list = np.vstack((shift_vector_list, self.find_shift_vector(i, frame)))
                else:
                    shift_vector_list = np.vstack((shift_vector_list, np.array([0, 0, 0])))

            max_shift_vector = self.find_max_shift_vector(shift_vector_list)
            # unit test
            print("the max shift vector for person {} is {}".format(i,max_shift_vector))
            self.broadcast_add_data_3d_std(i, max_shift_vector)  # broadcast add to self.data_3d_std

        return self.data_3d_std
