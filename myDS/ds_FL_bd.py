import copy

import numpy as np
import torch
from torchvision import datasets, transforms

from models.DeepS import DeepSight
from models.MnistBackdoor import BackdoorUpadte, Mnist_bd
from models.Net import CNNMnist
from models.Test import test_img, test_bd
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
        # split users
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
    mnist_bd = Mnist_bd(train=False)
    dict_attacker = mnist_iid(mnist_bd, int(args.num_users/2))
    # training
    global_model.train()
    loss_train = []
    for epoch in range(args.epochs):
        print("gobal epoch:{}".format(epoch))
        bd_loss = 0
        locals_loss = []
        seleted_users = max(int(args.frac * args.num_users), 1)
        if args.attack and epoch >= 5:
            seleted_users -= args.num_attacker
        users_idx = np.random.choice(range(args.num_users), seleted_users, replace=False)
        print("benign user", users_idx)

        # train local users model
        for idx in users_idx:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            loss = local.train_ds(net=copy.deepcopy(global_model).to(args.device), num=idx)
            locals_loss.append(copy.deepcopy(loss))
        if epoch >= 5:
            if args.attack:
                print("number of attacker:", args.num_attacker)
                for i in range(1, args.num_attacker + 1):
                    users_idx = np.append(users_idx, (-1 * i))
            for i in range(1, args.num_attacker + 1):
                bdu = BackdoorUpadte(args=args, idxs=dict_attacker[i])
                loss = bdu.bd_fa_ds(net=copy.deepcopy(global_model).to(args.device), num=(-1 * i))
                # locals_loss.append(copy.deepcopy(loss))
                bd_loss += loss
        print("all user", users_idx)

        # update global model
        w_agg = DeepSight(global_model=copy.deepcopy(global_model), user_list=users_idx, args=args)
        global_model.load_state_dict(w_agg)
        save_name = './save_model/model_ds_{}.pth'.format(epoch)
        torch.save(global_model.state_dict(), save_name)

        # print loss
        loss_avg = sum(locals_loss) / len(locals_loss)
        write_to_file(loss_avg, "ds_loss_main")
        write_to_file(bd_loss, "ds_loss_bd")
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)

        # testing
        global_model.eval()
        bd_acc_test = test_bd(global_model, mnist_bd, args)
        acc_train = test_img(global_model, dataset_train, args)
        acc_test = test_img(global_model, dataset_test, args)
        write_to_file(acc_test, "ds_acc_main")
        write_to_file(bd_acc_test, "ds_acc_bd")
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}\n".format(acc_test))
