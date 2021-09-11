import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from common.arguments import parse_args
from random import choice
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

from common.data_utils import *

class Visualization:
    '''
    obj Visualization:
        data: dict{
            'truth': dict{
                "view_0": dict{
                    "pose_c": array[n,x,17,3]
                    "pose_2d": array[n,x,17,2]
                    "camera": array[4]
                    "center": array[3]
                }
                "view_1"
            }
            'prediction': dict{
                "pose_pred": array[n,x,17,3]
                "trans": array[n,x,2]
                "center" (if calculated)
            }
        }
        info: dict{
            "args" args
            "frame" x(int)
            "number" n(int)
            "receptive_field" int, int
            "view_key" list of view
            "skeleton" list of connection
            "sample"
        }
    
    '''




    def __init__(self) -> None:

        self.data_truth = dict(); self.data_prediction = dict()
        self.info_args = parse_args()
        print(self.info_args)
        #load data
        
        
        Visualization.load_data(self)
        Visualization.get_datainfo(self)
        Visualization.process_prediction(self)
        

        #generate subplot

        Visualization.prepare_plot(self, self.info_args.compare)
        

        #tqdm
        if __name__ == '__main__':
            self.pbar = tqdm(total=self.info_frame)
        pass

    def load_data(self):
        try:
            if self.info_args.filepath is not None:
                filepath = self.info_args.file

            else:
                filepath = 'output/data_output_' + self.info_args.dataset + '_1.npz'
            #filepath = 'output/data_output_' + self.info_args.dataset + '_1.npz'
            self.info_filepath = filepath
            dataset_orig = np.load(filepath, allow_pickle=True)["positions_2d"].item()
            count_key_list = list(dataset_orig.keys())
            if self.info_args.sample is not None: 
                sample_key = int(self.info_args.sample); 
            else: 
                sample_key = random.choice( count_key_list )

            print("\n")
            print("Current File: " + filepath)
            print("Current Sample: " + str(sample_key))
            print("\n")
            
            self.data_truth = dataset_orig[sample_key]
            self.info_sample = sample_key

            prediction_orig = np.load(filepath, allow_pickle=True)["positions_3d"].item()
            prediction = prediction_orig[sample_key]
            
            pos_pred = prediction["pose_pred"].cpu().detach().numpy()
            #pos_trans = prediction["T"].cpu().detach().numpy()

            self.data_prediction["pose_pred"] = pos_pred
            #self.data_prediction["trans"] = pos_trans

            self.info_receptive_field = [prediction["receptive_field"],3] #list(prediction["receptive_field"])
            
        except KeyError:
            print('Sample does not exist! Please input right sample number')

        pass

    def process_prediction(self):
        """
        This module is used for scaling and changing coordinate
        """

        # Scaling
        self.data_prediction["pose_pred"] = self.data_prediction["pose_pred"] * 1e3

        # Change coordinate
        temp = np.copy(self.data_prediction["pose_pred"])
        self.data_prediction["pose_pred"][:,:,:,:,0] = - temp[:,:,:,:,0]
        self.data_prediction["pose_pred"][:,:,:,:,1] = - temp[:,:,:,:,2]
        self.data_prediction["pose_pred"][:,:,:,:,2] = - temp[:,:,:,:,1]

        temp = None

        self.data_prediction = Visualization.__get_center_1(self.data_prediction, "pose_pred", 1)


        for key in self.info_view_key:
            #self.data_truth[key]["pose_c"][:,:,:,:] = self.data_truth[key]["pose_c"][:,:,:,:] * 10 / 2
            print(self.data_truth[key]["pose_c"][:,:,:,:])
            temp = np.copy(self.data_truth[key]["pose_c"])
            self.data_truth[key]["pose_c"][:,:,:,2] = temp[:,:,:,1]
            self.data_truth[key]["pose_c"][:,:,:,1] = temp[:,:,:,2]

            self.data_truth[key] = Visualization.__get_center(self.data_truth[key], "pose_c", 1)

            pass

        pass

    def get_datainfo(self):
        
        dataset_metadata = suggest_metadata(self.info_args.dataset)
        try:
            self.info_view_key = list(self.data_truth.keys())
            self.info_number = self.data_truth[self.info_view_key[0]]['pose_2d'].shape[1]
            self.info_frame = self.data_prediction["pose_pred"].shape[2]
            self.info_skeleton = dataset_metadata['skeleton']
            self.info_num_joints = dataset_metadata['num_joints']
            #Visualization.cut_length(self, self.info_receptive_field[0], self.info_receptive_field[1])
        except KeyError:
            print("The dataset havent been fully supported yet")
        

        Visualization.cut_length(self, self.info_receptive_field[0], self.info_receptive_field[1])

        self.info_plot_radius = 4000
        

        
        pass

    def cut_length(self, receptive_field_1, receptive_field_2):
        start_true = int(receptive_field_1/2) - 1; end_true = start_true + self.info_frame
        start_pred = int(receptive_field_2/2) - 1; end_pred = start_pred + self.info_frame
    

        for key in self.info_view_key:
            self.data_truth[key]["pose_2d"] = self.data_truth[key]["pose_2d"][:,start_true:end_true,:,:]
            self.data_truth[key]["pose_c"] = self.data_truth[key]["pose_2d"][:,start_true:end_true,:,:]

        #self.data_prediction["trans"] = self.data_prediction["trans"][:,start_pred:end_pred,:]

        return

    def prepare_plot(self, ifcomp):

        self.info_color = []
        for i in range(self.info_number + 1): self.info_color.append("#" + "".join([choice("0123456789ABCDEF") for i in range(6)]))

        self.fig = plt.figure()
        if ifcomp:
            ax = dict()
            view_num = len(self.info_view_key) 
            i = 1
            for view_key in self.info_view_key:
                pos = np.array([i,i+1,i+2]) + 30 + view_num*100
                ax[view_key]=[self.fig.add_subplot(pos[0], projection='3d'), self.fig.add_subplot(pos[1]), self.fig.add_subplot(pos[2], projection='3d'),]

                ax[view_key][0].set_xlabel("x"); ax[view_key][0].set_xlabel("y"); ax[view_key][0].set_xlabel("z")
                ax[view_key][1].set_xlabel("x"); ax[view_key][1].set_xlabel("y"); 
                ax[view_key][2].set_xlabel("x"); ax[view_key][2].set_xlabel("y"); ax[view_key][2].set_xlabel("z")

                ax[view_key][0].set_title("Ground Truth: " + view_key); ax[view_key][1].set_title("Plane Image: " + view_key)
                ax[view_key][2].set_title("Prediction: " + view_key)
                i += 3
            self.ax = ax
        else:
            self.bx = self.fig.add_subplot(111, projection='3d')

            self.bx.set_xlabel("x"); self.bx.set_xlabel("y"); self.bx.set_xlabel("z")
            self.bx.set_title("Prediction")

        pass
        
    def plt2D(self, frame, ax, multiperson_data, camdata = None, ifclear = True, ifdot = False, ifscale = True):
        if ifclear: ax.clear()
        for k in range(multiperson_data.shape[0]):
            for i in self.info_skeleton:
                x = np.stack((multiperson_data[k, frame, i[0], 0], multiperson_data[k, frame, i[1], 0]), 0)
                y = np.stack((multiperson_data[k, frame, i[0], 1], multiperson_data[k, frame, i[1], 1]), 0)
                ax.plot(x, y, lw=2, c=self.info_color[k],alpha=0.6); 
            
            if ifdot:
                for j in range(17):
                    x = multiperson_data[k, frame, j, 0]
                    y = multiperson_data[k, frame, j, 1]
                    c = multiperson_data[k, frame, j, 2]

                    if c == 1:
                        ax.plot(x, y,'.',color='g',alpha=1)

                    if c == -1:
                        ax.plot(x,y,'.',color="r",alpha=1)

        if ifscale:
            fx = camdata[2]/2
            fy = camdata[3]/2
            cx = camdata[0]
            cy = camdata[1]    
            ax.set_xlim([cx-fx,cx+fx])
            ax.set_ylim([cy-fy,cy+fy])
        pass


    def plt3D(self, frame, ax, multiperson_data, center = None,
         dot = False, arrow = True, clear = True, ifscale = True, ifroot = True, iftrans = False, ifskeleton = True):
        if clear: ax.clear()
        
        for k in range(multiperson_data.shape[0]):
            
            if ifskeleton:
            
                for i in self.info_skeleton:

                    x = np.stack((multiperson_data[k, frame, i[0], 0], multiperson_data[k, frame, i[1], 0]), 0)
                    y = np.stack((multiperson_data[k, frame, i[0], 1], multiperson_data[k, frame, i[1], 1]), 0)
                    z = np.stack((multiperson_data[k, frame, i[0], 2], multiperson_data[k, frame, i[1], 2]), 0)

                    ax.plot3D(x, y, z, lw=2, c=self.info_color[k], alpha = 0.8); 
            
            if ifroot:
                root = multiperson_data[k, frame, 0, :]
                ax.plot3D(root[0], root[2], root[1], '.',color=self.info_color[k],alpha=1)
                label = '(%d, %d, %d)' % (root[0], root[2], root[1])
                ax.text(root[0], root[2], root[1], label)
            

            if dot:
                for j in range(17):
                    x = multiperson_data[k, frame, j, 0]
                    z = multiperson_data[k, frame, j, 1]
                    y = multiperson_data[k, frame, j, 2]

                    if j != 0: ax.plot3D(x, y, z, '.',color="r",alpha=1)

        if arrow:
            ax.arrow3D(0,0,0,
            0,1,0,
            mutation_scale=20,
            arrowstyle="-|>",
            linestyle='dashed')

        if ifscale:
            ax.set_xlim3d([center[0] - 2*self.info_plot_radius, center[0] + 2*self.info_plot_radius]) # 画布大小
            ax.set_ylim3d([center[1] - 2*self.info_plot_radius, center[1] + 2*self.info_plot_radius])
            ax.set_zlim3d([-self.info_plot_radius/4 + center[2], self.info_plot_radius + center[2]])
        pass


        
    def updater(self, frame):

        if self.info_args.compare:
            k = 0
            for key in self.info_view_key:
                # 2D Truth
                Visualization.plt2D(self, frame, self.ax[key][1], self.data_truth[key]["pose_2d"], self.data_truth[key]['camera'])
                # 3D Truth
                Visualization.plt3D(self, frame, self.ax[key][0], self.data_truth[key]["pose_c"], self.data_truth[key]["center"], False, True, True, True, False, False, True)
                # Prediction
                Visualization.plt3D(self, frame, self.ax[key][2] ,self.data_prediction["pose_pred"][k], self.data_prediction["center"][k], False, True, True, True, False, False, True)
                pass
            
        else:
            Visualization.plt3D(self, frame, self.bx, self.data_prediction["pose_pred"], self.data_prediction["center"],  True, True)

        if __name__ == '__main__':
            self.pbar.update(1)

        #input("Press any key to continue")

        pass


    def animate(self):
        '''
        Produce animation and save it
        '''
        anim = FuncAnimation(self.fig, self.updater, self.info_frame, interval=100000)
        plt.show()
        #if self.pb: anim.save("output/data_multi_output_" + self.ds + ".gif", writer='imagemagick')
        savepath = self.info_filepath[0:-4] + "_sample" + str(self.info_sample) + ".gif"
        if self.info_args.playback: anim.save(savepath, writer='pillow', fps=165)

        return

    def __get_center(data: dict, key, scale = 1):
        data['center'] = np.mean(data[key][:,0,0,:],axis=0) * scale
        return data
        
    def __get_center_1(data: dict, key, scale = 1):
        data['center'] = np.mean(data[key][:,:,0,0,:],axis=1) * scale
        return data

    def check_data(self, n):
        try:
            data = self.data_prediction["pose_pred"][n]
            for eachframe in data:
                try:
                    input("Press enter to continue")
                    print(eachframe)
                except SyntaxError:
                    pass
                except KeyboardInterrupt:
                    break
        except IndexError:
            print("index out of bound")
        pass

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
    #v.check_data(1)
    v.animate()


