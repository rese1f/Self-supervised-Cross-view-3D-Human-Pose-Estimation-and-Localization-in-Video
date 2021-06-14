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


class collision_eliminate:
    def __init__(self, data_cluster=None):
        # local variable initialization
        self.n = data_cluster.shape[0]
        self.x = data_cluster.shape[1]
        self.number_of_all_vertices = data_cluster.shape[2]
        self.data_cluster = data_cluster

        # intermediate variable initialization
        self.data_cluster_2d = torch.tensor([])
        self.bounding_box = torch.tensor([])
        self.central_distance_tensor = torch.tensor([])
        self.bounding_box_vertexbased = torch.tensor([])
        self.collision_cases_list = torch.tensor([])
        self.best_shift_vector_modified = torch.tensor([])

        self.con = ConfigParser()
        self.con.read('configs.ini')

    def test_bounding_box(self):
        test_path_list = ['../Dataexpand_dataset/S1_Directions.mat', '../Dataexpand_dataset/S1_Greeting.mat']
        # try animating the boxes inside 2 dimensions
        v_2d = vis_2d(data=self.data_cluster_2d, data_path_list=test_path_list, save_name='2d.gif')
        v_2d.animate()
        # compare with 3d case
        v_3d = vis(data=self.data_cluster, data_path_list=test_path_list, save_name='3d.gif')
        v_3d.animate()
        print("test_bounding_box: SUCCESS")
        return

    def throw_vertex_part(self):
        # bounding points are decided: [0, 12, 21, 18, 26, 29, 16, 3, 5, 8, 10]
        bounding_point = [int(i) for i in self.con.get('skeleton', 'bonding_point').split(',')]

        # initialize the tensor cube
        tensor_cube = []

        # iterate through n
        for data in self.data_cluster_2d:
            # data_cluster_2d has torch.Size([n, x, 32, 2])
            # data has torch.Size([x, 32, 2]), with n iteration rounds

            # initialize the tensor plane
            tensor_plane = []

            # iterate through x
            for frame in range(self.x):
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

        self.bounding_box = tensor_cube.permute(0, 2, 1)
        return self.bounding_box

    def find_2d_bounding_box(self):
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
        self.data_cluster_2d = torch.index_select(self.data_cluster, 3, indices)
        # print(data_cluster_2d.shape) gives torch.Size([2, x, 32, 2]), because z-axis is
        #   sliced out.

        # test if the 2D bounding box is successfully found
        # test_bounding_box(data_cluster_2d, data_cluster)

        # throw the vertex part
        self.bounding_box = self.throw_vertex_part()

        # return the bunch of bounding boxes
        return

    def find_central_distance(self):
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
        adjacency_matrix_tensor = torch.zeros(self.n, self.n, self.x)

        # find the central point tensor of bounding_box
        tensor_x = self.bounding_box[:, :, 0:2]
        tensor_y = self.bounding_box[:, :, 2:4]

        central_tensor_x = torch.mean(input=tensor_x, dim=2, keepdim=True)  # [n, x, 1 for x_center]
        central_tensor_y = torch.mean(input=tensor_y, dim=2, keepdim=True)  # [n, x, 1 for y_center]
        central_tensor_xy = torch.cat((central_tensor_x, central_tensor_y), dim=2)  # [n, x, 2 for (x,y)_center]

        # transform the central point tensor to the permutation of central distances [n, n, x]

        # find all the possibilities and store them into one list
        permutation_list = list(iter.combinations([i for i in range(self.n)], 2))  # [(0, 1), (0, 2), (1, 2)]

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

    def transfer_line_to_vertex_box(self):
        # 1----------2
        # |          |
        # |          |
        # 3----------4

        # localization
        x_max_tensor = self.bounding_box[:, :, 0]
        y_max_tensor = self.bounding_box[:, :, 1]
        x_min_tensor = self.bounding_box[:, :, 2]
        y_min_tensor = self.bounding_box[:, :, 3]

        # new order: x_min, y_max, x_max, y_max, x_min, y_min, x_max, y_min
        self.bounding_box_vertexbased = torch.stack((
            x_min_tensor, y_max_tensor, x_max_tensor, y_max_tensor,
            x_min_tensor, y_min_tensor, x_max_tensor, y_min_tensor
        ), dim=2)

        return

    def find_collision_cases(self):
        """Utility function for finding the

        After finding the distances between multiple boxes, it's time to find the
        candidate collision cases. We utilize the "candidate method", which is also
        utilized in visualize.py, to gain a result tensor of size [n, n, x].

        Args:
            bounding_box: The tensor of size [n, x, 4].
            bounding_box_vertexbased: The tensor of size [n, x, 8].

        Returns:
            All the collision cases, as a list of tensor of size [n, n, x]. Inside the tensor all values
            must be 0 (for no collision) and 1 (for collision)

        Raises:
            NOError: no error occurred up to now
        """

        # initialize the adjacency matrix tensor
        adjacency_matrix_tensor = torch.zeros(self.n, self.n, self.x)

        # find all the possibilities and store them into one list
        permutation_list = list(iter.combinations([i for i in range(self.n)], 2))  # [(0, 1), (0, 2), (1, 2)]

        # iterate through the list of [0, 1], [0, 2], [1, 2] and find the collision cases
        for permutation in permutation_list:

            # find the box range of both items
            x_range_max_0 = self.bounding_box[permutation[0], :, 0]
            y_range_max_0 = self.bounding_box[permutation[0], :, 1]
            x_range_min_0 = self.bounding_box[permutation[0], :, 2]
            y_range_min_0 = self.bounding_box[permutation[0], :, 3]

            x_range_max_1 = self.bounding_box[permutation[1], :, 0]
            y_range_max_1 = self.bounding_box[permutation[1], :, 1]
            x_range_min_1 = self.bounding_box[permutation[1], :, 2]
            y_range_min_1 = self.bounding_box[permutation[1], :, 3]

            # decide whether any vertex of one term is in the range of another term

            # any one vertex of four suffices
            for i in range(4):
                vertex_coordination_x_0 = self.bounding_box_vertexbased[permutation[0], :, i]
                vertex_coordination_y_0 = self.bounding_box_vertexbased[permutation[0], :, i + 1]

                vertex_coordination_x_1 = self.bounding_box_vertexbased[permutation[1], :, i]
                vertex_coordination_y_1 = self.bounding_box_vertexbased[permutation[1], :, i + 1]

                # iterate through each frame
                for frame in range(self.x):
                    # if the vertex is in the range
                    if (
                            (x_range_max_0[frame] > vertex_coordination_x_1[frame] > x_range_min_0[frame] and
                             y_range_max_0[frame] > vertex_coordination_y_1[frame] > y_range_min_0[frame]) or
                            (x_range_max_1[frame] > vertex_coordination_x_0[frame] > x_range_min_1[frame] and
                             y_range_max_1[frame] > vertex_coordination_y_0[frame] > y_range_min_1[frame])
                    ):
                        # mark the frame
                        adjacency_matrix_tensor[permutation][frame] = 1

        self.collision_cases_list = adjacency_matrix_tensor  # [n, n, x]
        return

    def find_shift_vector_candidate(self):
        """Auxiliary function for finding the shift vector pair

        We utilize only the vertical part of the shift vector, and choose 2 shift amounts for each
        shift request (each call to this function).

        Args:
            bounding_box_vertexbased: The tensor of size [n, x, 8].

        Returns:
            The result pair of candidate shift vectors (with whom it is acted on). Note that each vector contains
            both the VERTICAL shift amount and the item number to be shifted.

        Raises:
            NOError: no error occurred up to now
        """

        # we rule that the invalid point must be shifted to due north (upper) or due south (down)
        shift_vector_1 = torch.tensor([-1, 1])  # act on person no.1, go down 1 unit
        shift_vector_2 = torch.tensor([2, 2])  # act on person no.2, go up 2 units
        shift_vector = torch.stack([shift_vector_1, shift_vector_2])

        return shift_vector  # torch.Size([2, 2])

    def determine_shift_vectors_for_each_person(self, shift_vector_candidate_list):
        """Auxiliary function for finding the best shift vector for each person

        We now have the shift_vector_candidate_list of size [992, 2], and we'll iterate through it to find
        the three vectors with the largest absolute value to the 3 people respectively.

        Args:
            shift_vector_candidate_list: The tensor of size [992, 2]. For example, it can be of the form
            [-1.202, 2
             1.389,  0
             4.828,  2
             -4.618  1
              ...   ...]

        Returns:
            The unmodified best_shift_vector of size [3], respectively for person 0, 1, 2.

        Raises:
            NOError: no error occurred up to now
        """
        # divide all shift vectors for 3 people respectively
        if self.n > 6:
            print("PANIC. determine_shift_vectors_for_each_person() only allows at most 5 people.")
            exit()

        personal_shift_vector_candidate_list = [[], [], [], [], [], []]
        for person_number in range(self.n):
            # TEST CASES.
            # for person 0, we have shift 2.62, 3.17 (on frame 1654)
            # for person 1, we have shift -2.76, 1.16 (on frame 1476)
            # for person 2, we have shift 4.81, 2.75 (on frame 981)
            for counter in range(shift_vector_candidate_list.shape[0]):
                if shift_vector_candidate_list[counter, 1] == person_number:
                    personal_shift_vector_candidate_list[person_number].append\
                        (shift_vector_candidate_list[counter, 0])

        # pick 3 shift vectors for each person
        best_shift_vector = torch.tensor([0, 0, 0])
        for person_number in range(self.n):
            # decide the absolute value
            if personal_shift_vector_candidate_list[person_number] == []:  # no collisions
                best_shift_vector[person_number] = 0
            else:  # has collisions, find the shift with the maximum absolute value
                best_shift_value_positive = np.max(personal_shift_vector_candidate_list[person_number])
                best_shift_value_negative = np.min(personal_shift_vector_candidate_list[person_number])

                if best_shift_value_positive > -best_shift_value_negative:
                    best_shift_vector[person_number] = best_shift_value_positive.item() # from np.float to float
                else:
                    best_shift_vector[person_number] = best_shift_value_negative.item()

        # example: best_shift_vector = torch.tensor([1146, -2277, 1165])
        return best_shift_vector

    def decide_shift_vector(self):
        """Utility function for finding the perfect shift vector for all frames

        Firstly, we construct the shift vectors of all the collision cases, which
        contains 2 possibilities for each frame, as we want to gain a better realization
        of different direction of shifts. Secondly, we utilize a mathematical method on
        the list of all shift vectors to choose the vector with largest absolute value
        of length. This leads to a perfect match by conjecture.

        Args:
            collision_cases_list: the list containing collision case vectors, each of
                size [n, n, x].
            bounding_box_vertexbased: The tensor of size [n, x, 8].

        Returns:
            The answer of the best shift vector, which is an exact value.

        Raises:
            NOError: no error occurred up to now
        """

        # initialize the candidates for the shift vector
        shift_vector_candidate_list = torch.tensor([])

        # find all the possibilities and store them into one list
        permutation_list = list(iter.combinations([i for i in range(self.n)], 2))  # [(0, 1), (0, 2), (1, 2)]

        # iterate through the list of [0, 1], [0, 2], [1, 2] and find the collision cases
        for permutation in permutation_list:
            # iterate through each frame
            for frame in range(self.x):
                # if collision happens
                if self.collision_cases_list[permutation][frame] == 1:
                    # returns torch.Size[2, 2].
                    # e.g. [2.23 for shift, #1 for person
                    #      [-3.17 for shift, #2 for person]
                    shift_vector_candidate = self.find_shift_vector_candidate()
                    # add candidates into the list one by one
                    shift_vector_candidate_list = \
                        torch.cat((shift_vector_candidate_list, shift_vector_candidate), 0)

            # print(shift_vector_candidate_list.shape) gives torch.Size([992, 2])

        # determine the shift vector for each person
        best_shift_vector = self.determine_shift_vectors_for_each_person(shift_vector_candidate_list)

        # convert best_shift_vector to be best_shift_vector_modified of size [n, x, 32, 3]
        # e.g. [EFFECTED, EFFECTED, EFFECTED, 2 dimensions EFFECTED]
        self.best_shift_vector_modified = torch.zeros(self.n, self.x, self.number_of_all_vertices, 3)
        for person_number in range(self.n):
            # only shift in y direction
            self.best_shift_vector_modified[person_number, :, :, 1] = best_shift_vector[person_number]

        return self.best_shift_vector_modified

    def collision_eliminate(self):
        """Wrapper function for collision elimination

        Reconstruct content in the cluster of data, from [2, x, 32, 3] to [2, x_star, 32, 3]

        Args:
            data_cluster: The tensor of size [2, numOfFrames(or we call "x"), 32, 3].

        Returns:
            Literally returns nothing, but changes data_cluster in the memory heap directly.

        Raises:
            NOError: no error occurred up to now
        """

        self.find_2d_bounding_box()  # [n, x, 4]
        print("the shape of bounding_box is {}".format(self.bounding_box.shape))
        self.find_central_distance()
        print("the shape of distance tensor is {}".format(self.central_distance_tensor.shape))
        self.transfer_line_to_vertex_box()
        print("the shape of vertex-based bounding box is {}".format(self.bounding_box_vertexbased.shape))
        self.find_collision_cases()
        print("the shape of collision cases list is {}".format(self.collision_cases_list.shape))
        self.decide_shift_vector()
        print("the shape of best-shift vector (modified) is {}".format(self.best_shift_vector_modified.shape))

        self.data_cluster = self.data_cluster + self.best_shift_vector_modified
        return self.data_cluster  # nothing
