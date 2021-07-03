import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generating script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-min', '--min-object-number', default=2, type=int, metavar='MIN', help='min nunber of objects in new dataset')
    parser.add_argument('-max', '--max-object-number', default=4, type=int, metavar='MAX', help='max nunber of objects in new dataset')
    parser.add_argument('-s', '--shift', default=500, type=int, metavar='S', help='shift the timeline by the mean value S')
    parser.add_argument('-t', '--translation', default=1000, type=int, metavar='T', help='the mean translation distance')
    parser.add_argument('-r', '--rotation', default=True, type=bool, metavar='R', help='if rotate single raw data')
