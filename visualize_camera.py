import numpy as np
from configparser import ConfigParser
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import random_function as rf
import torch


class visualize_2d:
    def __init__(self, data=None, data_path_list=None, configs='configs.ini', radius=2000, if_box=True,
                 save_name='test.gif'):
        con = ConfigParser()
        con.read('Dataexpand-main\configs.ini')
        self.vertex_number = con.getint('skeleton', 'vertex_number')
        self.edge_number = con.getint('skeleton', 'edge_number')
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
        self.x0 = self.data[0][0, 0]
        self.y0 = self.data[0][0, 0]

        self.save_name = save_name

        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        if __name__ == '__main__':
            self.pbar = tqdm(total=self.frame)

    def update(self, frame: int):
        """
        Update frame
        """
        plt.cla()

        leftColor, rightColor, middleColor = "#3498db", "#e74c3c", "#000520"
        for data in self.data:
            for i in self.structure:
                x = torch.stack((data[frame, i[0], 0], data[frame, i[1], 0]), 0)
                y = torch.stack((data[frame, i[0], 2], data[frame, i[1], 2]), 0)
                if i[2] == 0:
                    self.ax.plot(x, y, lw=2, c=rightColor)
                elif i[2] == 1:
                    self.ax.plot(x, y, lw=2, c=leftColor)
                else:
                    self.ax.plot(x, y, lw=2, c=middleColor)

            
        
        self.ax.set_xlim([-5000, 5000])
        self.ax.set_ylim([-5000, 5000])
        
        if __name__ == '__main__':
            self.pbar.update(1)

        return
    '''
    def bonding_box(self, x_max, x_min, y_max, y_min, z_max, z_min):
        """
        Draw the bonding_box if need
        """
        bcolor = "#3ade70"
        vertex = torch.stack([
            torch.stack([x_max, y_max, z_max]),
            torch.stack([x_max, y_max, z_min]),
            torch.stack([x_max, y_min, z_max]),
            torch.stack([x_max, y_min, z_min]),
            torch.stack([x_min, y_max, z_max]),
            torch.stack([x_min, y_max, z_min]),
            torch.stack([x_min, y_min, z_max]),
            torch.stack([x_min, y_min, z_min]),
        ])
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
        for line in lines:
            x = torch.stack((line[0, 0], line[1, 0]), 0)
            y = torch.stack((line[0, 1], line[1, 1]), 0)
            z = torch.stack((line[0, 2], line[1, 2]), 0)
            self.ax.plot3D(x, y, z, lw=2, c=bcolor)

        return
    '''

    def animate(self):
        '''
        Produce animation and save it
        '''
        anim = FuncAnimation(self.fig, self.update, self.frame, interval=1)
        plt.show()
        # anim.save(self.save_name, writer='pillow', fps=165)

        return
