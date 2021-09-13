import numpy as np

phone = {
    'layout_name': 'phone',
    'height': [500,2000],
    'distance': [6000,7000],
    # if cross the center location
    'cross': False,
    # if head to center location
    'tracking': True,
    'inmat': np.array([[1149.7,     0,  508.8],
                       [     0,1147.6,  508.1],
                       [     0,     0,      1]],dtype=np.float32)
}

others = {

}

def suggest_metadata(name):
    names = []
    for metadata in [phone, others]:
        if metadata['layout_name'] in name:
            return metadata
        names.append(metadata['layout_name'])
    raise KeyError('Cannot infer camera layout from name "{}". Tried {}.'.format(name, names))