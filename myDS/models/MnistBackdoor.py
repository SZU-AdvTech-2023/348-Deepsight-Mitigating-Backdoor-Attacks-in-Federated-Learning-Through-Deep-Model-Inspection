import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Mnist_bd(Dataset):
    def __init__(self, trans=True, train=True):
        self.train = train
        self.trans = trans
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.transform = trans_mnist
        dataset = datasets.MNIST('../data/mnist/', train=self.train, download=True)
        image_bd = []
        target = []
        for image, label in dataset:
            img_arr0 = np.array(image)
            img_arr0[1:9, -9:-1] = 255
            image_bd.append(img_arr0)
            target.append(1)
            # img_arr1 = np.array(image)
            # image_bd.append(img_arr1)
            # target.append(label)
        self.target = target
        self.data = image_bd

        self.length = len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.target[index]

        if self.trans:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.length


class BackdoorUpadte(object):
    def __init__(self, args, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        dataset = Mnist_bd()
        self.local_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def bd_fa(self, net):
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

    def bd_fa_ds(self, net, num):
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
        name = './saved_updates/update_{}.pth'.format(num)
        torch.save(net.state_dict(), name)
        print("att loss", sum(epoch_loss) / len(epoch_loss))
        return sum(epoch_loss) / len(epoch_loss)
