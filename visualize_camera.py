import numpy as np
from configparser import ConfigParser
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import random_function as rf
import torch


class visualize_2d:
    def __init__(self, filename:str, configs='configs.ini', save_name='test_ac.gif'):
        con = ConfigParser()
        con.read('configs.ini')
        self.vertex_number = con.getint('skeleton', 'vertex_number')
        self.edge_number = con.getint('skeleton', 'edge_number')
        self.structure = np.reshape([int(i) for i in con.get('skeleton', 'structure').split(',')],(self.edge_number, 3))
        output_path = con.get('path', 'output_path')
        self.data = torch.tensor(scio.loadmat(output_path+filename)['data'])

        self.frame = np.min([self.data[i].shape[0] for i in range(len(self.data))])

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        
        self.save_name = save_name

        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


    def update(self, frame: int):
        """
        Update frame
        """
        plt.cla()

        for data in self.data:
            for i in self.structure:
                x = torch.stack((data[frame, i[0], 0], data[frame, i[1], 0]), 0)
                y = torch.stack((data[frame, i[0], 1], data[frame, i[1], 1]), 0)
                self.ax.plot(x, y, lw=2, color="b",alpha=0.2)
            
            for j in range(self.vertex_number):
                x = data[frame,j,0]
                y = data[frame,j,1]
                c = data[frame,j,2]

                if c == 0:
                    self.ax.plot(x, y,'.',color='g',alpha=1)

                if c == 1:
                    self.ax.plot(x,y,'.',color="r",alpha=1)


                    
        fx = 1527.4/2
        fy = 1529.2/2
        cx = 957.1
        cy = 529.8    
        
        self.ax.set_xlim([cx-fx,cx+fx])
        self.ax.set_ylim([cy-fy,cy+fy])
        

    def animate(self):
        '''
        Produce animation and save it
        '''
        anim = FuncAnimation(self.fig, self.update, self.frame, interval=1)
        plt.show()
        #anim.save(self.save_name, writer='pillow', fps=165)

        return


if __name__ == '__main__':
    filename = '1'
    v = visualize_2d(filename)
    v.animate()
