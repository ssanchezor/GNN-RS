import html
import re
from turtle import ht
import torch
import os
import pandas as pd
# model FM
from model_FM import FactorizationMachineModel

# model FM GCN
from model_GCN import FactorizationMachineModel_withGCN
from scipy.sparse import identity
from torch_geometric.utils import from_scipy_sparse_matrix

from train import testpartial, train_one_epoch
from build_dataset import CustomerArticleDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utilities import Popularity_Graphic



def table_string (list_rows): # generates html table from 2 rows  
    mask="<td>{}</td>"  
    table_s="<table>"
    for rowlist in list_rows:
        row="<tr>"
        for rowitem in rowlist:
            row+=mask.format (str(rowitem))
        row +="</tr>"
        table_s+=row
    table_s+= "<table>"
    return table_s
def image_path_from_item(dataset_path, item):
    s=f"{dataset_path}images/{item[0:3]}/{item}.jpg"
    s= "<img src=""{}"" width=""75"">".format(s)
    return s


def html_top_recomend(dataset_path, dic_map_items, \
        l_max_items, l_max_items_acum  ):

    lbl_item_list=[]; lbl_item_filenames=[]
    for idx, val in zip (l_max_items, l_max_items_acum): #
        # need sring verions for path with 0
        rl_item_string =('000000' + str(dic_map_items[idx]))[-10:]
        lbl_item_list.append(rl_item_string + "<br>" + str(idx) + "<br>" + str(val) ) # debug
        lbl_item_filenames.append(image_path_from_item(dataset_path,rl_item_string))        
    html_top_l=""
    while (len(lbl_item_list))>0:
        chk_items=lbl_item_list[0:10]
        chk_items_filenames=lbl_item_filenames[0:10]
        html_top_l+= table_string([chk_items_filenames, chk_items]) # historical
        lbl_item_list=lbl_item_list[10:]
        lbl_item_filenames=lbl_item_filenames[10:]
    # prepare data
    return html_top_l



def html_user_info(dataset_path, dic_map_users, dic_map_items, users_real, items_real, \
        df_iter, user, gt, l_recommened_list_user ):

    html_out=""
    gt_r =('000000' + str(dic_map_items[gt]))[-10:] # gt_real   
    lbl_gt_string=gt_r + "<br>" + str(gt) # debug " " + str(gt)
    #rec_list=[22130, 12372, 17968, 10083, 11771, 14861, 11959, 18271, 12576, 12390]
    lbl_user =users_real.loc[dic_map_users[user]]
    # first we need to recover user iteration 
    l_user_items_hist = [gt] + df_iter [df_iter.user == user]["item"].tolist()
    lbl_item_list=[]; lbl_item_filenames=[]    
    for idx in l_user_items_hist: # all transactions
        # need sring verions for path with 0
        rl_item_string =('000000' + str(dic_map_items[idx]))[-10:]
        lbl_item_list.append(rl_item_string + "<br>" + str(idx) ) # debug
        lbl_item_filenames.append(image_path_from_item(dataset_path,rl_item_string))        
    html_hist_l=""
    while (len(lbl_item_list))>0:
        chk_items=lbl_item_list[0:10]
        chk_items_filenames=lbl_item_filenames[0:10]
        html_hist_l+= table_string([chk_items_filenames, chk_items]) # historical
        lbl_item_list=lbl_item_list[10:]
        lbl_item_filenames=lbl_item_filenames[10:]
    # prepare data
    lbl_item_list=[]; lbl_item_filenames=[]    
    for idx in l_recommened_list_user: # recomment list user n
        # need sring verions for path with 0
        rl_item_string =('000000' + str(dic_map_items[idx]))[-10:]
        auxs=rl_item_string + "<br>" + str(idx)  # + str(idx) for debug debug            
        if gt==idx:
            auxs="<p style=""color:green;""><b>"+auxs +"</b></p"
        lbl_item_list.append(auxs ) 
        lbl_item_filenames.append(image_path_from_item(dataset_path,rl_item_string))

    # we generate user info        
    auxs="<p>User: {}</p>".format(lbl_user).replace("\n","<br>")
    html_out+=auxs        
    # debug html_out+="<p><b>Ground truth item</b></p>"
    # debug html_out+= table_string([[image_path_from_item(dataset_path,gt_r)],[lbl_gt_string]])
    html_out+="<p><b>Ground truth + last iteractions </b></p>"
    html_out+=html_hist_l
    html_out+="<p><b>Recommendations</b></p>"
    auxs=  table_string([lbl_item_filenames, lbl_item_list])
    html_out+=auxs

    return html_out




def info_model_report (model, dataset_path, res_info, \
                l_info, full_dataset, dict_recomen, title="None", topk=10):


    items_real = pd.read_csv("../data/articles.csv")
    items_real.set_index("article_id", inplace=True)
    users_real = pd.read_csv("../data/customers.csv")
    users_real.set_index("customer_id", inplace=True)

    aux_items= pd.read_csv("../data/article.dic", sep='\t',header=None)
    aux_users= pd.read_csv("../data/customer.dic",sep='\t',header=None)
    #pd_customers =  pd.read_csv("../data/customers.csv")
    #pd_articles =  pd.read_csv("../data/articles.csv")
    num_users= len(aux_users); num_items=len(aux_items)

    
    # need to recreate dictionaries
    # need to substract -1 to customer 
    # need to add customer to article
    # see build_dataset.py  / preprocess_items(self, data, num_customers):    
    dic_map_users={}; dic_map_items={}
    for index, row in aux_users.iterrows(): # 
        idx=row[1] # file has structure real cust \t index (1... num_cust)
        idx-=1
        if idx in dic_map_users:
            dic_map_users[idx]="Customer error"
        else:
            dic_map_users[idx]=row[0] # real customer
    # number of items in dictionary should be num_users
    assert num_users==len(dic_map_users), f"Number of users ({num_users} <> elements in dictionary {len(dic_map_users)}"

    for index, row in aux_items.iterrows(): # 
        idx=row[1] # file has structure real cust \t index (1... num_articles)
        idx+=num_users -1 # adds number user to in
        if idx in dic_map_items:
            dic_map_items[idx]="Item error"
        else:
            dic_map_items[idx]=row[0] # real customer
    # number of items in dictionary should be num_users
    assert num_items==len(dic_map_items), f"Number of items ({num_items} <> elements in dictionary {len(dic_map_items)}"
    # dos datasets 1 para artículos y otro para clientes
    # debug code verifies the dic
    for i in range (0,30): # para probar no sirve de nada -> debug con num_users-1):
        real_user =dic_map_users[i]
        resultado=users_real.loc[real_user] 
    for i in range (num_users+1,num_users+30): # para probar no sirve de nada -> debug con num_items)
        #real_item =('000000' + str(dic_map_items[i]))[-10:]
        real_item =dic_map_items[i]
        resultado=items_real.loc[real_item] 


    # in oder to display interactions article, we need a datafram of iterations
    l_iter =full_dataset.items.tolist()
    df_iter=pd.DataFrame(l_iter, columns=['user','item'])
    
    

    #mcanals dictionary for coverage. Number of articles in full dataset
    #  l_info=[l_users, l_gt_item,l_recommened_list, l_val_recommened_list, NDCG]
    l_users, l_gt_item,l_recommened_list, l_val_recommened_list, NDCG = l_info

    # report fist heading

    # Creating the HTML file
    # Adding the input data to the HTML file    
    model_name=str(model.__class__.__name__)
    html_out=""    
    html_out+='''<html>
    <head>
    <style>
    table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
    }
    </style>'''
    html_out=""    
    html_out+='''<html>
    <head>
    <style>
    table {
    border-collapse: separate;
    border-spacing: 5px;
    }
    table, th, td {
    border: 3px;
    }
    </style>'''
    html_head='''<title>{}</title>
    </head> 
    <body>
    <h1>{}</h1>
    <h2>Class name {}</h2>'''
    if title=="None":
        title=model_name
    html_head=html_head.format(title, title,model_name)
    html_out+=html_head
    html_out+= "<h2>Model results</h2>"
    html_out+=  table_string(res_info)

    
    html_out+= "<h2>Recommendation distribution</h2>"
    gragph_name="Report_{}".format(title.replace(" ",""))
    info= Popularity_Graphic(dict_recomen, gragph_name)
    fnamepng, info_l1,l_max_items, l_max_items_acum, tot_sum = info
    html_out+="<p>{}<p>".format(info_l1)
    html_out+= "<img src=""{}"">".format(fnamepng)
    html_out+="<h3>100 first items and accumulative percentage or 80% (whathever it happens first)</h3>"
    html_out+="<p>Total number of items -> {}<p>".format(tot_sum)
    aux_val=sum (l_max_items_acum)/tot_sum*100    
    html_out+="<p>Cummulative percentaje displayed items -> {} % <p>".format(f"{aux_val:.4f}")
    auxs= html_top_recomend(dataset_path, dic_map_items, l_max_items, l_max_items_acum  )
    html_out+= auxs
    html_out+= "<h2>First users sample</h2>"

    NUMBERFIRSTSAMPLES=30
    for n in range (0,NUMBERFIRSTSAMPLES):
        user = l_users[n] # user=3
        gt = l_gt_item[n] #  gt= 10083
        l_recommened_list_user=l_recommened_list[n]
        html_user= html_user_info (dataset_path,dic_map_users, dic_map_items, users_real, items_real,\
                 df_iter, user, gt, l_recommened_list_user )
        html_out+=html_user

    html_out+= "<p></p><h2>Top recomendations sample</h2>"
    NUMBERTOPSAMPLES=30
    nsample=NUMBERTOPSAMPLES
    for n in range (NUMBERFIRSTSAMPLES, num_users):        
        user = l_users[n] # user=3
        gt = l_gt_item[n] #  gt= 10083
        l_recommened_list_user=l_recommened_list[n]
        if gt == l_recommened_list_user[0]: # gt == first recomendatiosn
            html_user= html_user_info (dataset_path,dic_map_users, dic_map_items, users_real, items_real,\
                 df_iter, user, gt, l_recommened_list_user )
            html_out+=html_user
            nsample-=1
            if nsample==0:
                break
    html_out+= "<p></p><h2>Predictions without GT sample</h2>"
    NUMBERBADSAMPLES=30
    nsample=NUMBERBADSAMPLES
    for n in range (NUMBERFIRSTSAMPLES, num_users):        
        user = l_users[n] # user=3
        gt = l_gt_item[n] #  gt= 10083
        l_recommened_list_user=l_recommened_list[n]
        if gt not in l_recommened_list_user: # gt == first recomendatiosn
            html_user= html_user_info (dataset_path,dic_map_users, dic_map_items, users_real, items_real,\
                 df_iter, user, gt, l_recommened_list_user )
            html_out+=html_user
            nsample-=1
            if nsample==0:
                break

            



    auxs ="Report_{}.html".format(title.replace(" ",""))
    with open(auxs, 'w') as f:
        f.write(html_out)



   
    

if __name__ == '__main__':


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = "../data/"
    topk = 10


    full_dataset = CustomerArticleDataset(dataset_path, num_negatives_train=4, num_negatives_test=99)
    data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)


    #PATH="FactorizationMachineModel.pt"
    #model = FactorizationMachineModel(full_dataset.field_dims[-1], 32).to(device)

    PATH="FactorizationMachineModel_withGCN_epoch7.pt"
    attention = False
    identity_matrix = identity(full_dataset.train_mat.shape[0])
    identity_matrix = identity_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((identity_matrix.row, identity_matrix.col)).astype(np.int64))
    values = torch.from_numpy(identity_matrix.data)
    shape = torch.Size(identity_matrix.shape)
    identity_tensor = torch.sparse.FloatTensor(indices, values, shape)
    edge_idx, edge_attr = from_scipy_sparse_matrix(full_dataset.train_mat)
    model = FactorizationMachineModel_withGCN(full_dataset.field_dims[-1], 64, identity_tensor.to(device),
                                            edge_idx.to(device), attention).to(device)

    model.load_state_dict(torch.load(PATH, map_location=device))

    print ("Display test")
    hr, ndcg, cov, gini, dict_recomend, nov, l_info = testpartial(model, full_dataset, device, topk=topk)

    print(f'Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f}, COV@{topk} = {cov:.4f}, GINI@{topk} = {gini:.4f}, NOV@{topk} = {nov:.4f} ')


 
    res_header=[f"HR@{topk}", f"NDCG@{topk}", f"COV@{topk}",f"GINI@{topk}",f"NOV@{topk}" ]
    res_values=[f"{hr:.4f}", f"{ndcg:.4f}", f"{cov:.4f}", f"{gini:.4f}", f"{nov:.4f}"  ]
    res_info=[res_header,res_values]
    dataset_path = "../data/"

    info_model_report (model, dataset_path, res_info, l_info, \
            full_dataset, dict_recomend, title="Test report" , topk=10 )
    
