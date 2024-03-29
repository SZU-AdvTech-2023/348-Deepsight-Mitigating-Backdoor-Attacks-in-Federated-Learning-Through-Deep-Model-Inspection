import numpy as np


def mnist_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))  # 随机抽取
        all_idxs = list(set(all_idxs) - dict_users[i])  # 将被抽取的元素拿走
    return dict_users
