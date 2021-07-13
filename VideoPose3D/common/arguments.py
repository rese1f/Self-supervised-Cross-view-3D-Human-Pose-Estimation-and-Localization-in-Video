
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
    parser.add_argument('--save', default='trained_h36m_cpn.bin', type=str, metavar='FILENAME',
                        help='checkpoint to save (file name)')
    parser.add_argument('-v', '--multi-view', default=False, type=bool, metavar='V', help='if the dataset have multi-view')
    parser.add_argument('-eval', '--evaluate', default=True, type=bool, metavar='E', help='make evaluation if get 3d ground truth')
    parser.add_argument('-u', '--update', default=True, type=bool, metavar='U', help='if update the parameter of model')
    parser.add_argument('-o', '--output', default=False, type=bool, metavar='PATH',
                        help='output predict 3d pose')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')

    # Model arguments
    parser.add_argument('-s', '--stride', default=1, type=int, metavar='N', help='chunk size to use during training')
    parser.add_argument('-e', '--epochs', default=4, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-drop', '--dropout', default=0.25, type=float, metavar='P', help='dropout probability')
    parser.add_argument('-lrd', '--lr-decay', default=0.95, type=float, metavar='LR', help='learning rate decay per epoch')
    parser.add_argument('-arc', '--architecture', default='3,3,3,3,3', type=str, metavar='LAYERS', help='filter widths separated by comma')
    parser.add_argument('--causal', action='store_true', help='use causal convolutions for real-time processing')
    parser.add_argument('-ch', '--channels', default=1024, type=int, metavar='N', help='number of channels in convolution layers')

    # Experimental
    parser.add_argument('--dense', action='store_true', help='use dense convolutions instead of dilated convolutions')
    
    parser.set_defaults(bone_length_term=True)
    
    args = parser.parse_args()

    if not args.update and args.save:
        RuntimeError('cannot save model when stay constant')

    return args

def args_4_visc():
    parser = argparse.ArgumentParser(description='Training script')
    
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset') # h36m or humaneva
    parser.add_argument('-s', '--sample', default=0, type=int, metavar='NAME', help='sample number') 
    parser.add_argument('-p', '--playback', default=False, type=bool, metavar='NAME', 
        help='if saving the visualize result for playback') 
    parser.add_argument('-c', '--compare', default=True, type=bool, metavar='NAME', 
        help='if compare prediction with camera') 


    args = parser.parse_args()

    return args