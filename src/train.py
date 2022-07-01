import torch
import numpy as np
from statistics import mean
import math
import sys
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, criterion, device):
    model.train()
    total_loss = []
    for (interactions) in tqdm(data_loader, desc="EPOCH..."):
        interactions = interactions.to(device)
        targets = interactions[:, 2]
        predictions = model(interactions[:, :2])
        loss = criterion(predictions, targets.float())
        optimizer.zero_grad() # model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return mean(total_loss)


def train_one_epoch_features(model, optimizer, data_loader, criterion, device):
    model.train()
    total_loss = []
    for (interactions) in tqdm(data_loader, desc="EPOCH..."):
        interactions = interactions.to(device)
        targets = interactions[:, 3]
        predictions = model(interactions[:, :3])
        loss = criterion(predictions, targets.float())
        optimizer.zero_grad() # model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return mean(total_loss)


def testfull(model, full_dataset, device, topk=10):
    model.eval()
    HR, NDCG, NOVELTY = [], [], []
    num_items = full_dataset.field_dims[-1]
    num_customers = full_dataset.field_dims[0]
    dicitems = dict.fromkeys(list(range(num_customers, num_items)), 0)

    for user_test in full_dataset.test_set_AllItems:
        gt_item = user_test[0][1]
        novelty = 0
        predictions = model.predict(user_test, device)
        _, indices = torch.topk(predictions, topk)
        recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]

        HR.append(getHitRatio(recommend_list, gt_item))
        NDCG.append(getNDCG(recommend_list, gt_item))
        for item in recommend_list:
            if item in full_dataset.Popular_Items.index:
                rankItem = full_dataset.Popular_Items[item]
            else:
                rankItem = 0
            novelty += -math.log((rankItem / num_customers) + sys.float_info.epsilon, 2)  #
            if item in dicitems:
                dicitems[item] += 1
            else:
                dicitems[item] = 1
        NOVELTY.append(novelty / topk)
    gini = getGini(list(dicitems.values()))
    coverage = len({key: value for (key, value) in dicitems.items() if value > 0}) / (num_items - num_customers)
    return mean(HR), mean(NDCG), coverage, gini, dicitems, mean(NOVELTY)


def testpartial(model, full_dataset, device, topk=10):
    model.eval()
    num_items = full_dataset.field_dims[-1]
    num_customers = full_dataset.field_dims[0]
    HR, NDCG, NOVELTY = [], [], []
    dicitems = dict.fromkeys(list(range(num_customers, num_items)), 0)

    for user_test in full_dataset.test_set:
        gt_item = user_test[0][1]
        novelty = 0
        predictions = model.predict(user_test, device)
        _, indices = torch.topk(predictions, topk)
        recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]

        HR.append(getHitRatio(recommend_list, gt_item))
        NDCG.append(getNDCG(recommend_list, gt_item))

        for item in recommend_list:
            if item in full_dataset.Popular_Items.index:
                rankItem = full_dataset.Popular_Items[item]
            else:
                rankItem = 0
            novelty += -math.log((rankItem / num_customers) + sys.float_info.epsilon, 2)
            if item in dicitems:
                dicitems[item] += 1
            else:
                dicitems[item] = 1
        NOVELTY.append(novelty / topk)

    coverage = len({key: value for (key, value) in dicitems.items() if value > 0}) / (num_items - num_customers)
    gini = getGini(list(dicitems.values()))
    return mean(HR), mean(NDCG), coverage, gini, dicitems, mean(NOVELTY)


def getHitRatio(recommend_list, gt_item):
    if gt_item in recommend_list:
        return 1
    else:
        return 0


def getNDCG(recommend_list, gt_item):
    idx = np.where(recommend_list == gt_item)[0]
    if len(idx) > 0:
        return math.log(2)/math.log(idx+2)
    else:
        return 0


def getGini(x, w=None):
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n