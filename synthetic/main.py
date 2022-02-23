import torch
import numpy as np

if __name__ == '__main__':
    
    # Data Loader
    SOURCE_key = ['S1']
    TARGET_key = ['S5']
    
    DATA_3D = np.load('./data/data_3d_h36m.npz', allow_pickle=True)['positions_3d'].item()
    DATA_2D = np.load('./data/data_2d_h36m_gt.npz', allow_pickle=True)['positions_2d'].item()

    SOURCE, TARGET = [], []

    for subject in SOURCE_key:
        for action in DATA_3D[subject].keys():
            pair = (DATA_3D[subject][action], DATA_2D[subject][action])
            SOURCE.append(pair)
        
    for subject in TARGET_key:
        for action in DATA_3D[subject].keys():
            TARGET.append(DATA_2D[subject][action])
    
    