import copy
import math

import hdbscan
import numpy as np
import sklearn.metrics.pairwise as smp
import torch
from torch.utils.data import DataLoader, Dataset


class NoiseDataset(Dataset):
    def __init__(self, size, num_samples):
        self.size = size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.rand(self.size)
        return noise


def get_norm(model):
    squared_sum = 0
    for name, value in model.items():
        squared_sum += torch.sum(torch.pow(value, 2)).item()
    norm = math.sqrt(squared_sum)
    return norm


def dists_from_clust(clusters, num_users):
    pairwise_dists = np.ones((num_users, num_users))
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if clusters[i] == clusters[j] and clusters[i] != -1:
                pairwise_dists[i][j] = 0
    return pairwise_dists


def NEUPs(user_list, layer_name, gobal_params):
    neups, norm_list = [], []
    num_users = len(user_list)
    for i in range(len(user_list)):
        file_name = './saved_updates/update_{}.pth'.format(user_list[i])
        loaded_params = torch.load(file_name)
        norm = get_norm(loaded_params)
        gobal_norm = get_norm(gobal_params)
        norm_list = np.append(norm_list, norm - gobal_norm)
        bias_diff = abs(loaded_params['{}.bias'.format(layer_name)].cpu().numpy() - gobal_params[
            '{}.bias'.format(layer_name)].cpu().numpy())
        weight_diff_sum = np.sum(abs(loaded_params['{}.weight'.format(layer_name)].cpu().numpy() - gobal_params[
            '{}.weight'.format(layer_name)].cpu().numpy()), axis=1)
        ups = bias_diff + weight_diff_sum
        neup = ups ** 2 / np.sum(ups ** 2)
        neups = np.append(neups, neup)
    neups = np.reshape(neups, (num_users, -1))
    return neups, norm_list


def Threshold_Exceeding(neups, args):
    thresh_exds = []
    for neup in neups:
        te = 0
        for j in neup:
            if j >= (np.max(neup) / args.num_classes):
                te += 1
        thresh_exds.append(te)
    return thresh_exds


def Diffs(args, user_list, random_seeds, dim, global_model):
    local_model = copy.deepcopy(global_model)
    ddifs = []
    for seed in random_seeds:
        torch.manual_seed(seed)
        dataset = NoiseDataset([args.num_channels, dim, dim], args.num_samples)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        for i in range(len(user_list)):
            file_name = './saved_updates/update_{}.pth'.format(user_list[i])
            loaded_params = torch.load(file_name)
            local_model.load_state_dict(loaded_params)
            local_model.eval()
            global_model.eval()
            ddif = torch.zeros(args.num_classes).to(args.device)
            for data in loader:
                data = data.to(args.device)
                with torch.no_grad():
                    output_local = local_model(data)
                    output_global = global_model(data)
                tmp = torch.div(output_local, output_global + 1e-30)
                temp = torch.sum(tmp, dim=0)
                ddif.add_(temp)
            ddif /= args.num_samples
            ddifs = np.append(ddifs, ddif.cpu().numpy())
    ddifs = np.reshape(ddifs, (args.seed, len(user_list), -1))
    return ddifs


def DeepSight(global_model, user_list, args):
    # initialization
    w_agg = copy.deepcopy(global_model.state_dict())
    for k in w_agg.keys():
        w_agg[k] = 0
    num_users = len(user_list)
    tau = args.clipping_thresholds
    num_seeds = args.seed
    dim = args.dim
    layer_name = 'fc3'

    # filtering layer
    # cosine distance
    energy_sum = []
    for i in range(num_users):
        file_name = './saved_updates/update_{}.pth'.format(user_list[i])
        loacal_params = torch.load(file_name)
        loacal_bias = loacal_params['{}.bias'.format(layer_name)].cpu().numpy()
        global_bias = global_model.state_dict()['{}.bias'.format(layer_name)].cpu().numpy()
        energy_sum = np.append(energy_sum, loacal_bias - global_bias)
    cosine_dis = 1 - smp.cosine_distances(energy_sum.reshape(num_users, -1))

    # Threshold exceedings and NEUPs
    neups, norm_list = NEUPs(user_list=user_list, layer_name=layer_name, gobal_params=global_model.state_dict())
    thresh_exds = Threshold_Exceeding(neups=neups, args=args)
    print(thresh_exds)

    # random seeds
    random_seeds = []
    for i in range(args.seed):
        seed = int(np.random.rand() * 10000)
        random_seeds.append(seed)

    # ddif
    ddifs = Diffs(args=args, user_list=user_list, random_seeds=random_seeds, dim=dim, global_model=global_model)

    # classification
    classificat_boundary = np.median(thresh_exds)
    labels = []
    for i in thresh_exds:
        if i > classificat_boundary * 0.5:
            labels.append(False)
        else:
            labels.append(True)

    # clustering
    cosine_clusters = hdbscan.HDBSCAN(metric='precomputed').fit_predict(cosine_dis)
    cosine_cluster_dists = dists_from_clust(clusters=cosine_clusters, num_users=num_users)

    neup_clusters = hdbscan.HDBSCAN().fit_predict(neups)
    neup_cluster_dists = dists_from_clust(clusters=neup_clusters, num_users=num_users)

    ddif_cluster_dists = []
    for i in range(num_seeds):
        ddif_clusters = hdbscan.HDBSCAN().fit_predict(np.reshape(ddifs[i], (num_users, -1)))
        ddif_cluster_dist = dists_from_clust(clusters=ddif_clusters, num_users=num_users)
        ddif_cluster_dists = np.append(ddif_cluster_dists, ddif_cluster_dist)
    merged_ddif_cluster_dists = np.mean(np.reshape(ddif_cluster_dists, (num_seeds, num_users, num_users)), axis=0)

    # combine clusterings
    merged_distances = np.mean([merged_ddif_cluster_dists,
                                neup_cluster_dists,
                                cosine_cluster_dists], axis=0)
    clusters = hdbscan.HDBSCAN().fit_predict(merged_distances)
    print("clusters", clusters)

    # poisoned cluster identification
    positive_counts = {}
    total_counts = {}
    for i, cluster in enumerate(clusters):
        if cluster != -1:
            if cluster in positive_counts:
                positive_counts[cluster] += 1 if not labels[i] else 0
                total_counts[cluster] += 1
            else:
                positive_counts[cluster] = 1 if not labels[i] else 0
                total_counts[cluster] = 1

    # clipping and aggregation layer
    norm_threshold = np.median(norm_list)
    print("Clipping bound {}".format(norm_threshold))
    discard_cluster = []
    for i, c in enumerate(clusters):
        if c != -1:
            amount_of_positives = positive_counts[c] / total_counts[c]
            if amount_of_positives < tau:
                file_name = './saved_updates/update_{}.pth'.format(user_list[i])
                loaded_params = torch.load(file_name)
                if 1 > norm_threshold / norm_list[i]:
                    for name, data in loaded_params.items():
                        data.mul_(norm_threshold / norm_list[i])
                for name, value in loaded_params.items():
                    w_agg[name] += value
            else:
                discard_cluster.append(user_list[i])
        else:
            if labels[i]:
                discard_cluster.append(user_list[i])
            else:
                file_name = './saved_updates/update_{}.pth'.format(user_list[i])
                loaded_params = torch.load(file_name)
                if 1 > norm_threshold / norm_list[i]:
                    for name, data in loaded_params.items():
                        data.mul_(norm_threshold / norm_list[i])
                for name, value in loaded_params.items():
                    w_agg[name] += value
    print(discard_cluster)
    for k in w_agg.keys():
        w_agg[k] = torch.div(w_agg[k], len(user_list) - len(discard_cluster))
    return w_agg
