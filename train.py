import torch
import numpy as np
from statistics import mean
import math


def train_one_epoch(model, optimizer, data_loader, criterion, device):
    model.train()
    total_loss = []
    for i, (interactions) in enumerate(data_loader):
        interactions = interactions.to(device)
        targets = interactions[:, 2]
        predictions = model(interactions[:, :2])

        loss = criterion(predictions, targets.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    return mean(total_loss)


def test(model, full_dataset, device, topk=10):
    model.eval()
    HR, NDCG = [], []
    for user_test in full_dataset.test_set:
        gt_item = user_test[0][1]
        predictions = model.predict(user_test, device)
        _, indices = torch.topk(predictions, topk)
        recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]
        HR.append(getHitRatio(recommend_list, gt_item))
        NDCG.append(getNDCG(recommend_list, gt_item))
    return mean(HR), mean(NDCG)


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