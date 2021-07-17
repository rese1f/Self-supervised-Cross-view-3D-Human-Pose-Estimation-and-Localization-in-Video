import numpy as np
import matplotlib.pyplot as plt
import random
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

        self.args = parse_args()
        print(self.args)

        Visualization.load_data(self)
        Visualization.get_info(self)
        Visualization.subplot(self)



        #tqdm
        if __name__ == '__main__':
            self.pbar = tqdm(total=self.x)
        pass

    def load_data(self, ifdebug = False):
        try:
            dataset_orig = np.load('output/data_multi_' + self.args.dataset + '.npz', allow_pickle=True)["dataset"].item()
        except FileNotFoundError():
            print("Please select another dataset")

        #view_key = "view_" + str(args.view)
        try:
            
            if self.args.sample is not None:
                sample_key = int(self.args.sample)
            else:
                sample_key = random.choice(list(dataset_orig.keys()))
            self.dataset = dataset_orig[sample_key]
        except KeyError:
            #print('Sample does not exist! Please input right sample number')
            
            sample_key = random.choice(list(dataset_orig.keys()))
            self.dataset = dataset_orig[sample_key]
        pass

    def get_info(self):
        dataset_metadata = suggest_metadata(self.args.dataset)
        try:
            self.view_key_list = list(self.dataset.keys())
            self.n = self.dataset[self.view_key_list[0]]['pose_2d'].shape[0]
            self.x = self.dataset[self.view_key_list[0]]['pose_2d'].shape[1]
            self.skeleton = dataset_metadata['skeleton']
            self.joints_num = dataset_metadata['num_joints']
        except KeyError:
            print("The dataset havent been fully supported yet")
        self.radius = 4000
        self.center_c = Visualization.__get_center(self.dataset, self.view_key_list, 1e3)



        return


    def subplot(self):
        self.color = []
        for i in range(self.n): self.color.append("#" + "".join([choice("0123456789ABCDEF") for i in range(6)]))



        self.fig = plt.figure()
        ax = dict()
        view_num = len(self.view_key_list)
        i = 1
        for view_key in self.view_key_list:
            pos = np.array([i,i+1]) + 20 + view_num*100; 
            ax[view_key]=[self.fig.add_subplot(pos[0], projection='3d'), self.fig.add_subplot(pos[1])]
            ax[view_key][0].set_xlabel("x"); ax[view_key][0].set_xlabel("y"); ax[view_key][0].set_xlabel("z")
            ax[view_key][1].set_xlabel("x"); ax[view_key][1].set_xlabel("y")
            i += 2
        self.ax = ax
        pass

        
    def plt2D(self, ax, data, camera, frame, ifscale = True, ifclear = True, ifdot = True):
        if ifclear: ax.clear()
        multiperson_data = data; k = 0
        for person in multiperson_data:
            for i in self.skeleton:
                x = np.stack((person[frame, i[0], 0], person[frame, i[1], 0]), 0)
                y = np.stack((person[frame, i[0], 1], person[frame, i[1], 1]), 0)
                ax.plot(x, y, lw=2, c=self.color[k],alpha=0.6); 
            k += 1
            
            if ifdot:
                for j in range(17):
                    x = person[frame,j,0]
                    y = person[frame,j,1]
                    c = person[frame,j,2]

                    if c == 1:
                        ax.plot(x, y,'.',color='g',alpha=1)

                    if c == -1:
                        ax.plot(x,y,'.',color="r",alpha=1)

        if ifscale:
            camdata = camera            
            fx = camdata[2]/2
            fy = camdata[3]/2
            cx = camdata[0]
            cy = camdata[1]    
            ax.set_xlim([cx-fx,cx+fx])
            ax.set_ylim([cy-fy,cy+fy])
        pass


    def plt3D(self, ax, data, center, frame, ifarrow = True, ifclear = True, ifscale = True, transform = False):
        if ifclear: ax.clear()
        multiperson_data = data; k = 0
        for person in multiperson_data:
            for i in self.skeleton:
                x = np.stack((person[frame, i[0], 0], person[frame, i[1], 0]), 0)
                z = np.stack((person[frame, i[0], 1], person[frame, i[1], 1]), 0)
                y = np.stack((person[frame, i[0], 2], person[frame, i[1], 2]), 0)
                if transform:
                    temp = y; y = z; z = temp
                ax.plot3D(x, y, z, lw=2, c=self.color[k], alpha = 0.8); 
            k+=1

        if ifarrow:
            ax.arrow3D(0,0,0,
                0,1,0,
                mutation_scale=20,
                arrowstyle="-|>",
                linestyle='dashed')

        if ifscale:
            ax.set_xlim3d([center[0] - 2*self.radius, center[0] + 2*self.radius]) # 画布大小
            ax.set_ylim3d([center[2] - 2*self.radius, center[2] + 2*self.radius])
            ax.set_zlim3d([-self.radius/2 + center[1], self.radius/2 + center[1]])
        pass
        
    def updater(self, frame):

        for key in self.view_key_list:
            Visualization.plt2D(self, self.ax[key][1], self.dataset[key]["pose_2d"],
                self.dataset[key]['camera'], frame)
            Visualization.plt3D(self, self.ax[key][0], self.dataset[key]["pose_c"] * 1e3,
                self.center_c[key], frame, True, True)
            
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
        if self.args.playback: anim.save("player/output.gif", writer='pillow', fps=165)

        return

    def __get_center(data: dict, key_list, scale):
        center_dict = dict();
        for key in key_list:
            center_dict[key] = np.mean(data[key]["pose_c"][:,0,10,:],axis=0)*scale
            pass
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

