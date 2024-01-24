import copy

import numpy as np
import torch
from torchvision import datasets, transforms

from models.Fed import FedAvg
from models.MnistBackdoor import BackdoorUpadte, Mnist_bd
from models.Net import CNNMnist
from models.Test import test_img, test_bd
from models.Update import LocalUpdate
from utils.options import args_parser
from utils.sampling import mnist_iid


def write_to_file(num, fname):
    # 打开文件，以追加模式写入数据
    with open("{}.txt".format(fname), "a") as file:
        # 将epoch和loss写入文件
        file.write(f"{num}\n")


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    # sample users
    dict_users = mnist_iid(dataset_train, args.num_users)
    # build models
    global_model = CNNMnist().to(args.device)
    print(global_model)
    global_model.train()

    # copy weights
    w_glob = global_model.state_dict()
    mnist_bd = Mnist_bd(train=False)
    dict_attacker = mnist_iid(mnist_bd, int(args.num_users))

    # training
    w_locals = []
    sum_loss = []

    for epoch in range(args.epochs):
        print("gobal epoch:{}".format(epoch))
        bd_loss = 0
        locals_loss = []
        seleted_users = max(int(args.frac * args.num_users), 5)
        if args.attack and epoch >= 5:
            seleted_users -= args.num_attacker
        users_idx = np.random.choice(range(args.num_users), seleted_users, replace=False)
        print("benign user", users_idx)
        for idx in users_idx:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(global_model).to(args.device))
            w_locals.append(copy.deepcopy(w))
            locals_loss.append(copy.deepcopy(loss))
        # if epoch == 20:
        #     args.alr = 0.3
        # if epoch == 35:
        #     args.alr = 0.5
        # if epoch == 50:
        #     args.alr = 0.2
        if epoch >= 5:
            if args.attack:
                print("number of attacker:", args.num_attacker)
                for i in range(1, args.num_attacker + 1):
                    users_idx = np.append(users_idx, (-1 * i))
            for i in range(args.num_attacker):
                bdu = BackdoorUpadte(args=args, idxs=dict_attacker[i])
                w, loss = bdu.bd_fa(net=copy.deepcopy(global_model).to(args.device))
                w_locals.append(copy.deepcopy(w))
                bd_loss += loss
        print("all user", users_idx)
        bd_loss /= args.num_attacker
        # update global weights
        w_agg = FedAvg(w_locals)
        write_to_file(bd_loss, "bd_loss")
        # copy weight to net_glob
        global_model.load_state_dict(w_agg)
        save_name = './save_model/model_fa_bd_{}.pth'.format(epoch)
        torch.save(global_model.state_dict(), save_name)

        # print loss
        loss_avg = sum(locals_loss) / len(locals_loss)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        sum_loss.append(loss_avg)
        write_to_file(loss_avg, "main_loss")

        # testing
        global_model.eval()
        bd_acc_test = test_bd(global_model, mnist_bd, args)
        write_to_file(bd_acc_test, "bd_acc_test")
        acc_train = test_img(global_model, dataset_train, args)
        acc_test = test_img(global_model, dataset_test, args)
        write_to_file(acc_train, "main_acc_train")
        write_to_file(acc_test, "main_acc_test")
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}\n".format(acc_test))
