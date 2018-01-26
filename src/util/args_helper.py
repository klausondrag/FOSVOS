import argparse


def _get_base_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--gpu-id', default=None, type=int, help='The gpu id to use')

    parser.add_argument('--network', default='vgg16', type=str, choices=['vgg16', 'resnet18'],
                        help='The network to use')

    parser.add_argument('--no-training', action='store_true',
                        help='True if the program should train the model, else False')

    parser.add_argument('--no-testing', action='store_true',
                        help='True if the program should test the model, else False')

    parser.add_argument('--variant', default=None, type=int, help='version to try')

    return parser


def parse_args(is_online: bool) -> argparse.Namespace:
    parser = _get_base_parser()
    if is_online:
        parser.add_argument('-o', '--object', default='all', type=str, help='The object to train on')

    args = parser.parse_args()

    args.is_training = not args.no_training
    args.is_testing = not args.no_testing

    return args
