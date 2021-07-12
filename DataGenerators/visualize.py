import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from arguments import parse_args
from random import choice
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

from data_utils import *

class Visualization:
    def __init__(self) -> None:

        args = parse_args()
        print(args)

        #load data
        sample_key = int(args.number) - 1
        dataset_metadata = suggest_metadata(args.dataset)
        #view_key = "view_" + str(args.view)
        try:
            dataset_orig = np.load('output/data_multi_' + args.dataset + '.npz', allow_pickle=True)["dataset"].item()
            self.dataset = dataset_orig[sample_key]
        except KeyError:
            print('Sample does not exist! Please input right sample number')
        #print(self.dataset)
        #camera = dataset['camera']
        #self.dataset = dataset

        # get basic parameter
        #print(self.dataset)
        try:
            self.view_key_list = list(self.dataset.keys())
            self.n = self.dataset[self.view_key_list[0]]['pose_2d'].shape[0]
            self.x = self.dataset[self.view_key_list[0]]['pose_2d'].shape[1]
            self.skeleton = dataset_metadata['skeleton']
            self.joints_num = dataset_metadata['num_joints']
        except KeyError:
            print("The dataset havent been fully supported yet")
        self.radius = 4000
        self.center_c = Visualization.__get_center(self.dataset, self.view_key_list)



        #pre process



        # prepare color
        self.color = []
        for i in range(self.n): self.color.append("#" + "".join([choice("0123456789ABCDEF") for i in range(6)]))


        #generate subplot

        self.fig = plt.figure()
        ax = dict()
        view_num = len(self.view_key_list)
        i = 1
        for view_key in self.view_key_list:
            pos = np.array([i,i+1]) + 20 + view_num*100; print(pos)
            ax[view_key]=[self.fig.add_subplot(pos[0], projection='3d'), self.fig.add_subplot(pos[1])]
            ax[view_key][0].set_xlabel("x"); ax[view_key][0].set_xlabel("y"); ax[view_key][0].set_xlabel("z")
            ax[view_key][1].set_xlabel("x"); ax[view_key][1].set_xlabel("y")
            i += 2
        self.ax = ax

        #tqdm
        if __name__ == '__main__':
            self.pbar = tqdm(total=self.x)
        pass

        
    def plt2D(self, view_key, frame):
        self.ax[view_key][1].clear()
        multiperson_data = self.dataset[view_key]["pose_2d"] * 1e3; k = 0
        for person in multiperson_data:
            for i in self.skeleton:
                x = np.stack((person[frame, i[0], 0], person[frame, i[1], 0]), 0)
                y = np.stack((person[frame, i[0], 1], person[frame, i[1], 1]), 0)
                self.ax[view_key][1].plot(x, y, lw=2, c=self.color[k],alpha=0.6); 
            k += 1
            
            for j in range(17):
                x = person[frame,j,0]
                y = person[frame,j,1]
                c = person[frame,j,2]

                if c == 1:
                    self.ax[view_key][1].plot(x, y,'.',color='g',alpha=1)

                if c == -1:
                    self.ax[view_key][1].plot(x,y,'.',color="r",alpha=1)

        camdata = self.dataset[view_key]['camera']            
        fx = camdata[2]/2
        fy = camdata[3]/2
        cx = camdata[0]
        cy = camdata[1]    
        self.ax[view_key][1].set_xlim([cx-fx,cx+fx])
        self.ax[view_key][1].set_ylim([cy-fy,cy+fy])
        pass


    def plt3D(self, view_key, frame):
        self.ax[view_key][0].clear()
        multiperson_data = self.dataset[view_key]["pose_c"] * 1e3; k = 0
        for person in multiperson_data:
            for i in self.skeleton:
                x = np.stack((person[frame, i[0], 0], person[frame, i[1], 0]), 0)
                z = np.stack((person[frame, i[0], 1], person[frame, i[1], 1]), 0)
                y = np.stack((person[frame, i[0], 2], person[frame, i[1], 2]), 0)
                self.ax[view_key][0].plot3D(x, y, z, lw=2, c=self.color[k], alpha = 0.8); 
            k+=1

        self.ax[view_key][0].arrow3D(0,0,0,
           0,1,0,
           mutation_scale=20,
           arrowstyle="-|>",
           linestyle='dashed')


        self.ax[view_key][0].set_xlim3d([self.center_c[view_key][0] - 2*self.radius, self.center_c[view_key][0] + 2*self.radius]) # 画布大小
        self.ax[view_key][0].set_ylim3d([self.center_c[view_key][2] - 2*self.radius, self.center_c[view_key][2] + 2*self.radius])
        self.ax[view_key][0].set_zlim3d([-self.radius/2 + self.center_c[view_key][1], self.radius/2 + self.center_c[view_key][1]])
        pass

        
    def updater(self, frame):

        for key in self.view_key_list:
            Visualization.plt2D(self, key, frame)
            Visualization.plt3D(self, key, frame)
            
            pass

        if __name__ == '__main__':
            self.pbar.update(1)

        pass


    def animate(self):
        '''
        Produce animation and save it
        '''
        anim = FuncAnimation(self.fig, self.updater, self.x, interval=1)
        plt.show()
        #anim.save(self.save_name, writer='pillow', fps=165)

        return

    def __get_center(data: dict, key_list):
        center_dict = dict();
        for key in key_list:
            center_dict[key] = np.mean(data[key]["pose_c"][:,0,10,:],axis=0)*1e3
            pass
        print(center_dict)
        return center_dict

#   this prt used for arrow drawing
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)






#visualize main


if __name__ == '__main__':
    
    filename = '1'
    v = Visualization()
    v.animate()

