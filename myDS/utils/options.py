import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--alr', type=float, default=0.2, help="attacker learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    # models arguments
    parser.add_argument('--model', type=str, default='cnn', help='models name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--clipping_thresholds', type=float, default=1 / 3, help="clipping thresholds of update norm")
    parser.add_argument('--num_samples', type=int, default=20000, help="number of samples of noise dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--attack', action='store_true', help='backdoor attack')
    parser.add_argument('--num_attacker', type=int, default=1, help='number of attacker (default: 1)')
    parser.add_argument('--dim', type=int, default=28, help='noise dataset image size (same as training image)')

    args = parser.parse_args()
    return args
