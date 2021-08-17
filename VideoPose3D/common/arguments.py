
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset') # h36m or humaneva
    parser.add_argument('-k', '--keypoints_number', default=17, type=int, metavar='NAME', help='number of keypoints')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('-l', '--load', default='pretrained_h36m_cpn.bin', type=str, metavar='FILENAME',
                        help='checkpoint to load (file name)')

    # Model arguments
    parser.add_argument('-e', '--epochs', default=1, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-drop', '--dropout', default=0.25, type=float, metavar='P', help='dropout probability')
    parser.add_argument('-arc', '--architecture', default='3,3,3,3,3', type=str, metavar='LAYERS', help='filter widths separated by comma')
    parser.add_argument('--causal', action='store_true', help='use causal convolutions for real-time processing')
    parser.add_argument('-ch', '--channels', default=1024, type=int, metavar='N', help='number of channels in convolution layers')

    # Regressor arguments
    parser.add_argument('-w', '--width', default=16, type=int, metavar='N', 
                        help='number of frames in one shot (the more, the better, but is up to GPU Memory')
    parser.add_argument('--iter-nums', default=8, type=int, metavar='N',
                        help='iter_nums for regressor')

    # Experimental
    parser.add_argument('--dense', action='store_true', help='use dense convolutions instead of dilated convolutions')
    
    # Visualize arguments
    parser.add_argument('-ns', '--sample', default=None, type=int, metavar='NAME', help='sample number') 
    parser.add_argument('-p', '--playback', default=False, type=bool, metavar='NAME', 
        help='if saving the visualize result for playback') 
    parser.add_argument('-cp', '--compare', default=True, type=bool, metavar='NAME', 
        help='if compare prediction with gt') 

    args = parser.parse_args()

    return args