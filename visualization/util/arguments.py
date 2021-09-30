
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # Dataset arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset') # h36m or humaneva
    parser.add_argument('-k', '--keypoints_number', default=17, type=int, metavar='NAME', help='number of keypoints')
    
    # IO arguments
    #parser.add_argument('-ns', '--sample', default=None, type=int, metavar='NAME', help='sample number') 
    parser.add_argument('-i', '--inputpath', default=None, type=str, metavar='NAME', 
        help='if loading from specific path')
    parser.add_argument('-o', '--outputpath', default="data\output", type=str, metavar='NAME', 
        help='if loading from specific path') 

    args = parser.parse_args()

    return args