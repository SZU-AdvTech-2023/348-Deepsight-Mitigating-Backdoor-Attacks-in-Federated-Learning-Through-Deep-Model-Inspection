import copy

import torch
from torchvision import datasets, transforms

from models.MnistBackdoor import BackdoorUpadte, Mnist_bd
from models.Net import CNNMnist
from models.Test import test_bd
from models.Test import test_img
from models.Update import LocalUpdate
from utils.options import args_parser
from utils.sampling import mnist_iid

args = args_parser()
args.device = torch.device('cuda:0')

global_model = CNNMnist().to(args.device)
trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
print(global_model)
global_model.train()
w_locals = []

dict_users = mnist_iid(dataset_train, args.num_users)
mnist_bd = Mnist_bd(train=False)
dict_attacker = mnist_iid(mnist_bd, args.num_users)
bdu = BackdoorUpadte(args=args, idxs=dict_attacker[0])
# iddx=dict_users[0]+dict_users[1]+dict_users[2]+dict_users[0]
idxx = dict_attacker[0].union(dict_attacker[1])
local = LocalUpdate(args=args, dataset=dataset_train, idxs=idxx)

for i in range(30):
    print(i)
    w, loss2 = bdu.bd_fa(net=copy.deepcopy(global_model).to(args.device))
    global_model.load_state_dict(w)
    test_bd(global_model, mnist_bd, args)
    test_img(global_model, dataset_train, args)

test_img(global_model, dataset_train, args)
test_bd(global_model, mnist_bd, args)
