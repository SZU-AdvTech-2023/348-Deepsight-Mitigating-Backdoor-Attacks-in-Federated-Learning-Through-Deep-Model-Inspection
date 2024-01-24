import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net, datatest, args):
    net.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    for data, labels in data_loader:
        data, labels = data.to(args.device), labels.to(args.device)
        log_probs = net(data)
        test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(datatest)
    accuracy = 100.00 * correct / len(datatest)
    print('Test set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(datatest), accuracy))
    return accuracy


def test_bd(net, datatest, args):
    net.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    for data, labels in data_loader:
        data, labels = data.to(args.device), labels.to(args.device)
        log_probs = net(data)
        test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(datatest)
    accuracy = 100.00 * correct / len(datatest)
    print('BD Test set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(datatest), accuracy))
    return accuracy
