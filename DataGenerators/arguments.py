import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generating script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-min', '--min', default=3, type=int, metavar='MIN', help='min nunber of objects in new dataset')
    parser.add_argument('-max', '--max', default=5, type=int, metavar='MAX', help='max nunber of objects in new dataset')
    parser.add_argument('-s', '--shift', default=500, type=int, metavar='S', help='the mean value shift the timeline by')
    parser.add_argument('-t', '--translation', default=1000, type=int, metavar='T', help='the mean translation distance')
    parser.add_argument('-r', '--rotation', default=True, type=bool, metavar='R', help='if rotate single raw data')

    # Generating arguments
    parser.add_argument('-n', '--number', default=128, type=int, metavar='N', help='number of the generated dataset')
    parser.add_argument('-c', '--camera', default=['phone'], type=list, metavar='C', help='the type of camera used')
    parser.add_argument('-v', '--view', default=1, type=int, metavar='V', help='the number of view')

    # Visualize arguments
    parser.add_argument('-sp', '--sample', default=None, type=int, metavar='SP', help='the sample number u want 2 display')
    parser.add_argument('-p', '--playback', default=True, type=bool, metavar='PB', help='if saving as gif 4 playback')

    args = parser.parse_args()

    if args.view != len(args.camera):
        raise KeyError('view number does not match camera number')

    return args