import numpy as np

phone = {
    'layout_name': 'phone',
    'height': [500,2000],
    'distance': [2000,5000],
    # if cross the center location
    'cross': False,
    # if head to center location
    'tracking': True,
    'inmat': np.array([[1527.4,     0,  957.1],
                       [     0,1529.2,  529.8],
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