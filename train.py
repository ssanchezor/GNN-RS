import torch
import numpy as np
from statistics import mean
import math
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, criterion, device):
    model.train()
    total_loss = []
    #for i, (interactions) in enumerate(data_loader):
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

def test(model, full_dataset, device, topk=10):
    
    model.eval()
    #mcanals dictionary for coverage. Number of articles in full dataset
    num_items= full_dataset.field_dims[-1]
    num_customers=  full_dataset.field_dims[0]
    coverage=[]
    
    HR, NDCG = [], []
    #for user_test in full_dataset.test_set:
    for (user_test) in tqdm (full_dataset.test_set, desc= "Eval test dataset: "):
        gt_item = user_test[0][1] # mcanals positive target 
        predictions = model.predict(user_test, device) #mcanals models return 100 predicitons
        _, indices = torch.topk(predictions, topk) #mcanals 10 values with better scores
        recommend_list = user_test[indices.cpu().detach().numpy()][:, 1] #mcanals items recommended
        HR.append(getHitRatio(recommend_list, gt_item)) #mcanals our already positive 
        NDCG.append(getNDCG(recommend_list, gt_item))
        coverage.append(recommend_list[:topk])
        
    cov = len(np.unique(coverage))/(num_items-num_customers)
    return mean(HR), mean(NDCG), cov


def test_Rand_WRONG (model, full_dataset, device, topk=10):
    #mcanals dictionary for coverage. Number of articles in full dataset
    num_items= full_dataset.field_dims[-1]
    num_customers=  full_dataset.field_dims[0]
    dicitems={}
    
    HR, NDCG = [], []
    #for user_test in full_dataset.test_set:
    for (user_test) in tqdm (full_dataset.test_set, desc= "Eval test dataset: "):
        gt_item = user_test[0][1] # mcanals positive target 
        recommend_list = model.create(gt_item, topk) 
        HR.append(getHitRatio(recommend_list, gt_item)) #mcanals our already positive 
        NDCG.append(getNDCG(recommend_list, gt_item))
        for gt_item in recommend_list:
            if gt_item in dicitems:
                dicitems[gt_item]+=1
            else:
                dicitems[gt_item]=1
    coverage = len(dicitems)/(num_items-num_customers)
    # suma= sum(dicitems.values()) # deberia ser num customer x topk 
    return mean(HR), mean(NDCG), coverage



def getHitRatio(recommend_list, gt_item):
    if gt_item in recommend_list:
        return 1
    else:
        return 0


def getNDCG(recommend_list, gt_item):
    #mcanals Normalized Discounted Cumulative Gain
    #mcanals how relevant the results are and how good the ordering is?
    # list has tpk     
    idx = np.where(recommend_list == gt_item)[0] #mcanals position of gt_item 
    #mcanals p0=math.log(2)/math.log(0+2) # 1
    #mcanals p1=math.log(2)/math.log(1+2) # 0.63
    #mcanals p2=math.log(2)/math.log(2+2) # 0.5
    #mcanals p3=math.log(2)/math.log(3+2) # 0.43
    #mcanals p4=math.log(2)/math.log(4+2) # 0.38
    #mcanals p5=math.log(2)/math.log(5+2) # 0.35
    if len(idx) > 0:
        return math.log(2)/math.log(idx+2)
    else:
        return 0