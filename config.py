""" Config class for training the InvNet """

import argparse

def get_parser(name):
    """

    :param name: String for Config Name
    :return: parser
    """

    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser

class InvNetConfig(argparse.Namespace):
    def build_parser(self):
        parser = get_parser("InvNet config")
        parser.add_argument('--dataset', default='mnist', help='circle / polycrystalline')
        parser.add_argument('--trainset_path', default='/Users/kellymarshall/PycharmProjects/Invnet/datasets/circle/train_toyCircle_3Ch_128.h5',
                            help='Train dataset path')
        parser.add_argument('--validset_path', default='/Users/kellymarshall/PycharmProjects/Invnet/datasets/circle/valid_toyCircle_3Ch_128.h5',
                            help='Valid dataset path')
        parser.add_argument('--lr',default=01e-04)
        parser.add_argument('--output_path', default='./output_dir', help='output directory')
        parser.add_argument('--data_dir', default='data/MNIST')
        parser.add_argument('--gpu', default=0, help='Selecting the gpu')
        parser.add_argument('--batch_size', default=32,type=int, help='Batch size for training')
        parser.add_argument('--hidden_size', default=32, type=int,help='Hidden size used for generator and discriminator')
        parser.add_argument('--critic_iter', default=5, type=int,help='Number of iter for descriminator')
        parser.add_argument('--proj_iter', default=5, type=int, help='Number of iteration for projection update.')
        parser.add_argument('--end_iter', default=100000, help='How many iterations to train for.')
        parser.add_argument('--lambda_gp', default=10, help='gradient penalty hyperparameter')
        parser.add_argument('--restore_mode', default=False,
                            help='If True, it will load saved model from OUT_PATH and continue to train')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

class TestConfig(argparse.Namespace):
    def build_parser(self):
        parser = get_parser("InvNet config")
        parser.add_argument('--dataset', required=False, default='mnist', help='circle / polycrystalline')
        parser.add_argument('--trainset_path', default='/Users/kellymarshall/PycharmProjects/Invnet/datasets/circle/train_toyCircle_3Ch_128.h5',
                            help='Train dataset path')
        parser.add_argument('--validset_path', default='/Users/kellymarshall/PycharmProjects/Invnet/datasets/circle/valid_toyCircle_3Ch_128.h5',
                            help='Valid dataset path')
        parser.add_argument('--lr',default=01e-04)
        parser.add_argument('--output_path', required=False,default='./nada/', help='output directory')
        parser.add_argument('--data_dir',default='/Users/kellymarshall/PycharmProjects/graph_invnet/files/')
        parser.add_argument('--gpu', default=0, help='Selecting the gpu')
        parser.add_argument('--batch_size', default=2, help='Batch size for training')
        parser.add_argument('--hidden_size', default=4, help='Hidden size used for generator and discriminator')
        parser.add_argument('--critic_iter', default=1, help='Number of iter for descriminator')
        parser.add_argument('--proj_iter', default=1, help='Number of iteration for projection update.')
        parser.add_argument('--end_iter', default=1, help='How many iterations to train for.')
        parser.add_argument('--lambda_gp', default=10, help='gradient penalty hyperparameter')
        parser.add_argument('--restore_mode', default=False,
                            help='If True, it will load saved model from OUT_PATH and continue to train')
        parser.add_argument('--dp_loss_sign', default=-1,
                     help='Sign of the loss optimized during projection update')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
