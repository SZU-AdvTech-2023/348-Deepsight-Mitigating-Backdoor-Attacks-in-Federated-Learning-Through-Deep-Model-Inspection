import copy

import numpy as np
import torch
from torchvision import datasets, transforms

from models.Fed import FedAvg
from models.Net import CNNMnist
from models.Test import test_img
from models.Update import LocalUpdate
from utils.options import args_parser
from utils.sampling import mnist_iid


def write_to_file(num, fname):
    with open("{}.txt".format(fname), "a") as f:
        f.write(f"{num}\n")


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        dict_users = mnist_iid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build models
    if args.model == 'cnn' and args.dataset == 'mnist':
        global_model = CNNMnist().to(args.device)
    else:
        exit('Error: unrecognized models')
    print(global_model)
    global_model.train()

    # copy weights
    w_glob = global_model.state_dict()

    # training
    w_locals = []
    for epoch in range(args.epochs):
        print("gobal epoch:{}".format(epoch))
        locals_loss = []
        seleted_users = max(int(args.frac * args.num_users), 5)
        if args.attack and epoch >= 4:
            seleted_users -= args.num_attacker
        users_idx = np.random.choice(range(args.num_users), seleted_users, replace=False)
        print("benign user", users_idx)
        for idx in users_idx:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(global_model).to(args.device))
            w_locals.append(copy.deepcopy(w))
            locals_loss.append(copy.deepcopy(loss))

        # update global weights
        w_agg = FedAvg(w_locals)

        # copy weight to net_glob
        global_model.load_state_dict(w_agg)
        save_name = './save_model/model_fa_{}.pth'.format(epoch)
        torch.save(global_model.state_dict(), save_name)

        # print loss
        loss_avg = sum(locals_loss) / len(locals_loss)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        locals_loss.append(loss_avg)
        # testing
        global_model.eval()
        acc_train = test_img(global_model, dataset_train, args)
        acc_test = test_img(global_model, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}\n".format(acc_test))
        write_to_file(acc_test, "fa_acc")
