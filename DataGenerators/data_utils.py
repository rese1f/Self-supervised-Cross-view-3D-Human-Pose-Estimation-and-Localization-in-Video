import numpy as np

# for huamn3.6m, we choose the 17 joints of the total 32 joints
h36m_metadata = {
    'layout_name': 'h36m',
    'num_joints': 32,
    'keypoints': [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27],
    'skeleton': [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]],
}

others = {
    
}

def suggest_metadata(name):
    names = []
    for metadata in [h36m_metadata, others]:
        if metadata['layout_name'] in name:
            return metadata
        names.append(metadata['layout_name'])
    raise KeyError('Cannot infer keypoint layout from name "{}". Tried {}.'.format(name, names))