import torch
import numpy as np
from statistics import mean
import math
import sys
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, criterion, device):
    # trains the model
    model.train() # sets model to train status
    total_loss = []
    for (interactions) in tqdm(data_loader, desc="EPOCH..."):
        interactions = interactions.to(device) # extracts input
        targets = interactions[:, 2] # desired output
        predictions = model(interactions[:, :2]) # computes predicted output
        loss = criterion(predictions, targets.float()) # computes loss
        optimizer.zero_grad() #  sets gradients to zero
        loss.backward() # computes gradients
        optimizer.step() # updates parameters
        total_loss.append(loss.item())

    return mean(total_loss)


def train_one_epoch_context(model, optimizer, data_loader, criterion, device):
    # trains the model (considering context)
    model.train() # sets model to train status
    total_loss = []
    for (interactions) in tqdm(data_loader, desc="EPOCH..."):
        interactions = interactions.to(device) # extracts input
        targets = interactions[:, 3] # desired output
        predictions = model(interactions[:, :3]) # computes predicted output
        loss = criterion(predictions, targets.float()) # computes loss
        optimizer.zero_grad() # sets gradients to zero
        loss.backward() #computes gradients
        optimizer.step() # updates parameters
        total_loss.append(loss.item())

    return mean(total_loss)


def testfull(model, full_dataset, device, topk=10):
    # tests the model versus full test dataset
    model.eval() # sets model to test status
    HR, NDCG, NOVELTY = [], [], []
    num_items = full_dataset.field_dims[-1]
    num_customers = full_dataset.field_dims[0]
    dicitems = dict.fromkeys(list(range(num_customers, num_items)), 0)

    # evaluates for full test dataset
    for user_test in full_dataset.test_set_AllItems:
        gt_item = user_test[0][1]
        novelty = 0
        predictions = model.predict(user_test, device)
        _, indices = torch.topk(predictions, topk)
        recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]
        HR.append(getHitRatio(recommend_list, gt_item)) # computes HR metric
        NDCG.append(getNDCG(recommend_list, gt_item)) # computes NDCG metric
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
        NOVELTY.append(novelty / topk) # computes Novelty metric
    gini = getGini(list(dicitems.values())) # computes GINI metric
    coverage = len({key: value for (key, value) in dicitems.items() if value > 0}) / (num_items - num_customers) # computes Coverage metric
    return mean(HR), mean(NDCG), coverage, gini, dicitems, mean(NOVELTY)


def testpartial(model, full_dataset, device, topk=10, features=False):
    # tests the model versus partial test dataset
    model.eval() # sets model to test status
    num_items = full_dataset.field_dims[-1]
    num_customers = full_dataset.field_dims[0]
    HR, NDCG, NOVELTY = [], [], []
    dicitems = dict.fromkeys(list(range(num_customers, num_items)), 0)

    # set up of visualization lists
    l_users=[]
    l_gt_item=[]
    l_recommened_list=[]
    l_val_recommened_list=[]
    if features:
        l_gt_channel=[]

    # evaluates for partial test dataset
    for (user_test) in tqdm (full_dataset.test_set, desc= "PARTIAL Eval test dataset: "): # debug de mcanals
        gt_item = user_test[0][1]
        if features:
            gt_channel=user_test[0][2]
        novelty = 0
        predictions = model.predict(user_test, device)
        _, indices = torch.topk(predictions, topk)
        recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]
        HR.append(getHitRatio(recommend_list, gt_item)) # computes HR metric
        NDCG.append(getNDCG(recommend_list, gt_item)) # computes NDCG metric
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
        NOVELTY.append(novelty / topk) # computes Novelty metric

        # visualization
        l_gt_item.append(gt_item)
        l_users.append(user_test[0][0])        
        l_recommened_list.append(recommend_list)
        if features:
            l_gt_channel.append(gt_channel) # positive item
            l_info=[l_users, l_gt_item,l_gt_channel, l_recommened_list, l_val_recommened_list, NDCG]
        else:
            l_info=[l_users, l_gt_item,l_recommened_list, l_val_recommened_list, NDCG]

    coverage = len({key: value for (key, value) in dicitems.items() if value > 0}) / (num_items - num_customers) # computes Coverage metric
    gini = getGini(list(dicitems.values())) # computes GINI metric
    return mean(HR), mean(NDCG), coverage, gini, dicitems, mean(NOVELTY) , l_info


def getHitRatio(recommend_list, gt_item):
    # calculates Hit Ratio metric (fraction of users for which the correct article is included in the recommendation list)
    if gt_item in recommend_list:
        return 1
    else:
        return 0


def getNDCG(recommend_list, gt_item):
    # calculates NDGC metric (takes into consideration order of the recommended articles)
    idx = np.where(recommend_list == gt_item)[0]
    if len(idx) > 0:
        return math.log(2)/math.log(idx+2)
    else:
        return 0


def getGini(x, w=None):
    # calculates GINI coefficient (statistical dispersion of the provided recommendations)
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