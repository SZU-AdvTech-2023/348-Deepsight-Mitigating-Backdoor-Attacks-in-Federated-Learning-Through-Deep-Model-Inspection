import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Dataset_flips(Dataset):  # 用于标签翻转攻击
    def __init__(self, task, trans=True, train=True):
        self.train = train
        self.trans = trans
        if task == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            self.transform = trans_mnist
            dataset_flips = datasets.MNIST('../data/mnist/', train=self.train, download=True)
            self.data = [data for data, target in dataset_flips if target == 7]
        else:
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.transform = trans_cifar
            dataset_flips = datasets.CIFAR10('../data/cifar', train=self.train, download=True)
            self.data = [data for data, target in dataset_flips if target == 7]

        self.length = len(self.data)

    def __getitem__(self, index):
        # 获取数据和标签
        img = self.data[index]
        target = 1  # 将所有标签设置为1

        if self.trans:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.length


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.local_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train_ds(self, net, num):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for i in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.local_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        name = './saved_updates/update_{}.pth'.format(num)
        torch.save(net.state_dict(), name)
        return sum(epoch_loss) / len(epoch_loss)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for i in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.local_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class FlipsUpdate(object):
    def __init__(self, args):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        dataset = Dataset_flips(task=args.dataset)
        self.local_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)

    def flips_ds(self, net, num):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.alr, momentum=self.args.momentum)
        epoch_loss = []
        for i in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.local_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                batch_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        name = './saved_updates/update_{}.pth'.format(num)
        torch.save(net.state_dict(), name)
        print("att loss", sum(epoch_loss) / len(epoch_loss))
        return sum(epoch_loss) / len(epoch_loss)

    def flips_fa(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.alr, momentum=self.args.momentum)
        epoch_loss = []
        for i in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.local_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print("att loss", sum(epoch_loss) / len(epoch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
