import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from arguments import parse_args

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
        multiperson_data = self.dataset[view_key]["pose_2d"] * 1e3
        for person in multiperson_data:
            for i in self.skeleton:
                x = np.stack((person[frame, i[0], 0], person[frame, i[1], 0]), 0)
                y = np.stack((person[frame, i[0], 1], person[frame, i[1], 1]), 0)
                self.ax[view_key][1].plot(x, y, lw=2, color="b",alpha=0.2)
            
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
        multiperson_data = self.dataset[view_key]["pose_c"] * 1e3
        for person in multiperson_data:
            for i in self.skeleton:
                x = np.stack((person[frame, i[0], 0], person[frame, i[1], 0]), 0)
                z = np.stack((person[frame, i[0], 1], person[frame, i[1], 1]), 0)
                y = np.stack((person[frame, i[0], 2], person[frame, i[1], 2]), 0)
                self.ax[view_key][0].plot3D(x, y, z, lw=2, c="#000520", alpha = 0.3)

        self.ax[view_key][0].set_xlim3d([self.center_c[view_key][0] - 2*self.radius, self.center_c[view_key][0] + 2*self.radius]) # 画布大小
        self.ax[view_key][0].set_ylim3d([self.center_c[view_key][2] - 2*self.radius, self.center_c[view_key][2] + 2*self.radius])
        self.ax[view_key][0].set_zlim3d([0, self.radius + self.center_c[view_key][1]])
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










#visualize


if __name__ == '__main__':
    filename = '1'
    v = Visualization()
    v.animate()

