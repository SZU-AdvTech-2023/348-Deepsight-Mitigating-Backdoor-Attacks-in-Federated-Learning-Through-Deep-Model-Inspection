import torch
from models.Net import CNNMnist
from models.Test import test_bd
from models.Update import Dataset_flips
from utils.options import args_parser

args = args_parser()
args.device = torch.device('cuda:0')
net = CNNMnist().to(args.device)

net.load_state_dict(torch.load('./save_model/model_ds_{}.pth'.format(5)))
datatest = Dataset_flips(task=args.dataset, train=False)
test_bd(net, datatest, args)
