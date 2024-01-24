import copy
import torch


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for i in range(1, len(w)):
        for k in w_avg.keys():
            w_avg[k] += w[i][k]
    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
