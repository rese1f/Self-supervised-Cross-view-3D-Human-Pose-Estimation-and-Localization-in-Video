import numpy as np
from configparser import ConfigParser
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import torch


class visualize_collision:
    def __init__(self, data=None, data_path_list=None, configs='configs.ini', radius=2000, if_box=True,
                 save_name='test_2d.gif'):
        """
        ctor for vis_2d(visualize_2d)
        """
        con = ConfigParser()
        con.read('configs.ini')
        self.vertex_number = con.getint('skeleton', 'vertex_number')
        self.edge_number = con.getint('skeleton', 'edge_number')

        # read the structure in config.ini
        self.structure = np.reshape([int(i) for i in con.get('skeleton', 'structure').split(',')],
                                    (self.edge_number, 3))

        if data_path_list is not None:
            self.data = [torch.tensor(scio.loadmat(i)['data']) for i in data_path_list]
            self.data = [data.reshape(data.shape[0], self.vertex_number, 3) for data in self.data]

        if data is not None:
            self.data = data

        self.frame = np.min([self.data[i].shape[0] for i in range(len(self.data))])

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.radius = radius
        self.x0 = self.data[0][0, 0, 1]
        self.y0 = self.data[0][0, 0, 1]
        self.if_box = if_box

        # read config.ini, get the candidates for the bounding points
        # get a list
        self.bounding_point = [int(i) for i in con.get('skeleton', 'bonding_point').split(',')]

        self.save_name = save_name

        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        if __name__ == '__main__':
            self.pbar = tqdm(total=self.frame)

    def update(self, frame: int):
        """
        Update frame
        """
        plt.cla()

        for data in self.data:
            # candidate vertices for the box
            if self.if_box:
                x_sus = [data[frame, i, 0] for i in self.bounding_point]
                y_sus = [data[frame, i, 1] for i in self.bounding_point]

                # switch into the tensor
                x_sus = torch.stack(x_sus, dim=0)
                y_sus = torch.stack(y_sus, dim=0)
                x_max = torch.max(x_sus)
                x_min = torch.min(x_sus)
                y_max = torch.max(y_sus)
                y_min = torch.min(y_sus)

                # find the bounding_box
                self.bounding_box(x_max, x_min, y_max, y_min)

        self.ax.set_xlim(self.x0 - 2*self.radius, self.x0 + 2*self.radius)  # 画布大小
        self.ax.set_ylim(self.y0 - 2*self.radius, self.y0 + 2*self.radius)

        if __name__ == '__main__':
            self.pbar.update(1)

        return

    def bounding_box(self, x_max, x_min, y_max, y_min):
        """
        Draw the bounding_box if needed
        """

        bcolor = "#3ade70"

        # vertices of the rectagular box with 8 vertices (using a tensor)
        # use numpy can be better
        vertex = torch.stack([
            torch.stack([x_max, y_max]),
            torch.stack([x_max, y_max]),
            torch.stack([x_max, y_min]),
            torch.stack([x_max, y_min]),
            torch.stack([x_min, y_max]),
            torch.stack([x_min, y_max]),
            torch.stack([x_min, y_min]),
            torch.stack([x_min, y_min])
        ])

        # stack the lines
        lines = torch.stack([
            torch.stack([vertex[0], vertex[2]]),
            torch.stack([vertex[0], vertex[4]]),
            torch.stack([vertex[1], vertex[5]]),
            torch.stack([vertex[1], vertex[3]]),
            torch.stack([vertex[2], vertex[3]]),
            torch.stack([vertex[2], vertex[6]]),
            torch.stack([vertex[3], vertex[7]]),
            torch.stack([vertex[4], vertex[5]]),
            torch.stack([vertex[4], vertex[6]]),
            torch.stack([vertex[5], vertex[7]]),
            torch.stack([vertex[6], vertex[7]]),
            torch.stack([vertex[0], vertex[1]]),
        ])

        # draw the lines
        for line in lines:
            x = torch.stack((line[0, 0], line[1, 0]), 0)
            y = torch.stack((line[0, 1], line[1, 1]), 0)
            self.ax.plot(x, y, lw=2, c=bcolor)

        return

    # 平移完后不要有碰撞
    # 先判断碰撞后进行平移
    # 先平移再判断再平移

    def animate(self):
        """
        Produce animation and save it
        """
        anim = FuncAnimation(self.fig, self.update, self.frame, interval=1)
        plt.show()
        # print("now saving...")

        # if save
        # anim.save(self.save_name, writer='pillow', fps=165)

        return
