import numpy as np

# for huamn3.6m, we choose the 17 joints of the total 32 joints
h36m_metadata = {
    'layout_name': 'h36m',
    'num_joints': 32,
    'keypoints': [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27],
    'skeleton': [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]],
    'connect_' : [None,  6,    15,   None, 3,    12,   9,    18,   27,   45,    42,    48,     54,     39,    51,     57    ],
    'connect' : {
        0:  [0,7], 
        1:  [4,5],
        2:  [1,2],
        3:  [0,0],
        4:  [5,6],
        5:  [2,3],
        6:  [7,8],
        7:  [0,0],
        8:  [0,0],
        9:  [8,9],
        10: [0,0],
        11: [0,0],
        12: [0,0],
        13: [8,14],
        14: [8,11],
        15: [9,10],
        16: [11,12],
        17: [14,15],
        18: [12,13],
        19: [15,16],
        20: [0,0],
        21: [0,0],
        22: [0,0],
        23: [0,0],
    },
    'tree_connect' : {
        0:  [ 0,  1,  2  ],
        1:  [ 4,  5,  6  ],
        2:  [ 9,  13, 14 ],
        3:  [ 15, 16, 17 ],
        4:  [ 17, 18, 19 ],
    }
}

others = {
    
}

star_model = {
    'layout_name': 'star_model',
    'standard': [[0,0,1],[0,0,-1],[0,0,-1],[0,0,1],[0,0,-1],[0,0,-1],[0,0,1],[0,-1,0],[0,-1,0],[0,0,1],[0,-1,0],[0,-1,0],[0,0,1],
                 [1,0,0],[-1,0,0],[0,0,1],[1,0,0],[-1,0,0],[1,0,0],[-1,0,0],[1,0,0],[-1,0,0],[1,0,0],[-1,0,0]
    ],
}


def suggest_metadata(name):
    names = []
    for metadata in [h36m_metadata, star_model]:
        if metadata['layout_name'] in name:
            return metadata
        names.append(metadata['layout_name'])
    raise KeyError('Cannot infer keypoint layout from name "{}". Tried {}.'.format(name, names))