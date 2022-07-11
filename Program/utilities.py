import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pareto


def Popularity_Graphic(total_recommendations_dict, Title):
    # generates custom graphs representing long-tail (distribution of ratings or popularity among articles in the total dataset).
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(8, 4))

    plt.title(r'Item Recommendation ' + Title)
    plt.xlabel('Items ranked by popularity')
    plt.ylabel('Number of recommendations')
    dict_order = dict(sorted(total_recommendations_dict.items(), key=lambda item: item[1], reverse=True))
    list_dict = list(dict_order.values())
    print('max num recommendations: ', list_dict[0])
    list_index = list(dict_order.keys())
    list_accumsum = np.cumsum(list_dict)

    total_recommendations = list_accumsum[len(list_accumsum) - 1]
    print('Total Recommendations: ', total_recommendations)
    z = 0
    for i in list_accumsum:
        if i >= 0.8 * total_recommendations:
            break
        else:
            z = z + 1
    # 100 items or 0.8 at least for report
    if z>=100:
        l_max_items=list_index[0:100]
        l_max_items_acum=list_dict[0:100]
    else:
        l_max_items=list_index[0:z]
        l_max_items_acum=list_dict[0:z]
    tot_sum=sum(list_dict)

    print(Title)
    info_l1= "Number Products at what reach the 80% of the recommendations {} (Accumulate recommendations {})".format(z, list_accumsum[z])
    print(info_l1)
    print('Products:{0:2.2f}% '.format((z / len(list_dict)) * 100))
    print(list_index[:10])
    head_tail_split = z
    plt.plot(range(head_tail_split), list_dict[:head_tail_split], alpha=0.8, label=r'Head tail')
    plt.plot(range(head_tail_split, len(list_index)), list_dict[head_tail_split:], label=r'Long tail')
    plt.plot(head_tail_split, list_dict[head_tail_split], '-gD', label='80% of recommendations done')
    plt.axhline(y=list_dict[head_tail_split], linestyle='--', lw=1, c='grey')
    plt.axvline(x=head_tail_split, linestyle='--', lw=1, c='grey')
    plt.xlim([-5, len(list_index)])
    plt.ylim([-5, list_dict[0]])
    plt.legend()
    plt.tight_layout()
    fname = "../data/" + Title + ".pdf"
    fnamepng = "../data/" + Title + ".png"
    plt.savefig(fname)
    plt.savefig(fnamepng)
    info=[]
    info.append(fnamepng)
    info.append(info_l1)
    info.append(l_max_items)
    info.append(l_max_items_acum)
    info.append(tot_sum)
    return info

    

def table_string (list_rows): # generates html table from 2 rows  
    mask="<td>{}</td>"  
    table_s="<table>"
    for rowlist in list_rows:
        row="<tr>"
        for rowitem in rowlist:
            row+=mask.format (str(rowitem))
        row +="</tr>\r\n"
        table_s+=row
    table_s+= "</table>\r\n"
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
            auxs="<p style=""color:green;""><b>"+auxs +"</b></p>"
        lbl_item_list.append(auxs ) 
        lbl_item_filenames.append(image_path_from_item(dataset_path,rl_item_string))

    # we generate user info        
    auxs="<p>User: {}</p>\r\n".format(lbl_user).replace("\n","<br>\r\n")
    html_out+=auxs        
    # debug html_out+="<p><b>Ground truth item</b></p>"
    # debug html_out+= table_string([[image_path_from_item(dataset_path,gt_r)],[lbl_gt_string]])
    html_out+="<p><b>Ground truth + last iteractions </b></p>\r\n"
    html_out+=html_hist_l
    html_out+="<p><b>Recommendations</b></p>\r\n"
    auxs=  table_string([lbl_item_filenames, lbl_item_list])
    html_out+=auxs

    return html_out


def html_user_info_with_feature(dataset_path, dic_map_users, dic_map_items, dic_map_channels,  users_real, items_real, \
        df_iter, user, gt, gt_channel,  l_recommened_list_user ):

    html_out=""
    gt_r =('000000' + str(dic_map_items[gt]))[-10:] # gt_real   
    lbl_gt_string=gt_r + "<br>" + str(gt) # debug " " + str(gt)
    #rec_list=[22130, 12372, 17968, 10083, 11771, 14861, 11959, 18271, 12576, 12390]
    lbl_user =users_real.loc[dic_map_users[user]]
    # first we need to recover user iteration 
    l_user_items_hist = [gt] + df_iter [df_iter.user == user]["item"].tolist()
    l_user_items_hist_channel =[gt_channel] + df_iter [df_iter.user == user]["channel"].tolist()
    lbl_item_list=[]; lbl_item_filenames=[]; lbl_user_items_features=[]
    for idx,idx_feature in zip( l_user_items_hist,l_user_items_hist_channel): # all transactions
        # need sring verions for path with 0
        rl_item_string =('000000' + str(dic_map_items[idx]))[-10:]
        lbl_item_list.append(rl_item_string + "<br>" + str(idx) ) # debug
        lbl_item_filenames.append(image_path_from_item(dataset_path,rl_item_string))        
        lbl_user_items_features.append("channel "+ str(dic_map_channels[idx_feature]))
    html_hist_l=""
    while (len(lbl_item_list))>0:
        chk_items=lbl_item_list[0:10]
        chk_items_filenames=lbl_item_filenames[0:10]
        chk_items_channels=lbl_user_items_features[0:10]
        html_hist_l+= table_string([chk_items_filenames, chk_items,chk_items_channels]) # historical
        lbl_item_list=lbl_item_list[10:]
        lbl_item_filenames=lbl_item_filenames[10:]
        lbl_user_items_features=lbl_user_items_features[10:]

    # prepare data 
    # there is no channel in recommendation
    lbl_item_list=[]; lbl_item_filenames=[]
    for idx in l_recommened_list_user: # recomment list user n
        # need sring verions for path with 0
        rl_item_string =('000000' + str(dic_map_items[idx]))[-10:]
        auxs=rl_item_string + "<br>" + str(idx)  # + str(idx) for debug debug            
        auxs2=str(dic_map_items[idx])
        if gt==idx:
            auxs="<p style=""color:green;""><b>"+auxs +"</b></p>"
        lbl_item_list.append(auxs ) 
        lbl_item_filenames.append(image_path_from_item(dataset_path,rl_item_string))

    # we generate user info        
    auxs="<p>User: {}</p>\r\n".format(lbl_user).replace("\n","<br>")
    html_out+=auxs        
    # debug html_out+="<p><b>Ground truth item</b></p>"
    # debug html_out+= table_string([[image_path_from_item(dataset_path,gt_r)],[lbl_gt_string]])
    html_out+="<p><b>Ground truth + last iteractions </b></p>\r\n"
    html_out+=html_hist_l
    html_out+="<p><b>Recommendations</b></p>\r\n"
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
    html_out+= "<h2>Model results</h2>\r\n"
    html_out+=  table_string(res_info)

    
    html_out+= "<h2>Recommendation distribution</h2>\r\n"
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
    html_out+= "<h2>First users sample</h2>\r\n"

    NUMBERFIRSTSAMPLES=20
    for n in range (0,NUMBERFIRSTSAMPLES):
        user = l_users[n] # user=3
        gt = l_gt_item[n] #  gt= 10083
        l_recommened_list_user=l_recommened_list[n]
        html_user= html_user_info (dataset_path,dic_map_users, dic_map_items, users_real, items_real,\
                 df_iter, user, gt, l_recommened_list_user )
        html_out+=html_user

    html_out+= "<p></p><h2>Top recomendations sample</h2>\r\n"
    NUMBERTOPSAMPLES=5
    nsample=NUMBERTOPSAMPLES
    for n in range (NUMBERFIRSTSAMPLES+1, num_users):        
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
    html_out+= "<p></p><h2>Predictions without GT sample</h2>\r\n"
    NUMBERBADSAMPLES=5
    nsample=NUMBERBADSAMPLES
    for n in range (NUMBERFIRSTSAMPLES+1, num_users):        
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

def info_model_report_with_feature (model, dataset_path, res_info, \
                l_info, full_dataset, dict_recomen, title="None", topk=10):


    items_real = pd.read_csv("../data/articles.csv")
    items_real.set_index("article_id", inplace=True)
    users_real = pd.read_csv("../data/customers.csv")
    users_real.set_index("customer_id", inplace=True)
    # no channels_real as this is a feature embedded in the transaction

    aux_items= pd.read_csv("../data/article.channel.dic", sep='\t',header=None)
    aux_users= pd.read_csv("../data/customer.channel.dic",sep='\t',header=None)
    aux_channels= pd.read_csv("../data/channel.channel.dic",sep='\t',header=None)
    #pd_customers =  pd.read_csv("../data/customers.csv")
    #pd_articles =  pd.read_csv("../data/articles.csv")
    num_users= len(aux_users); num_items=len(aux_items); num_channels=len(aux_channels)

    
    # need to recreate dictionaries
    # need to substract -1 to customer 
    # need to add customer to article
    # see build_dataset.py  / preprocess_items(self, data, num_customers):    
    dic_map_users={}; dic_map_items={}; dic_map_channels={}; 
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

    for index, row in aux_channels.iterrows(): # 
        idx=row[1] # file has structure real cust \t index (1... num_articles)
        idx+=num_users + num_items-1 # adds number users and items to idx
        if idx in dic_map_items:
            dic_map_channels[idx]="Item error"
        else:
            dic_map_channels[idx]=row[0] # real customer
    # number of items in dictionary should be num_users
    assert num_channels==len(dic_map_channels), f"Number of channels ({num_channels} <> elements in dictionary {len(dic_map_channels)}"
    



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
    df_iter=pd.DataFrame(l_iter, columns=['user','item','channel'])
    
    

    #mcanals dictionary for coverage. Number of articles in full dataset
    #  l_info=[l_users, l_gt_item,l_recommened_list, l_val_recommened_list, NDCG]
    l_users, l_gt_item,l_gt_channel,  l_recommened_list, l_val_recommened_list, NDCG = l_info

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

    NUMBERFIRSTSAMPLES=20
    for n in range (0,NUMBERFIRSTSAMPLES):
        user = l_users[n] # user=3
        gt = l_gt_item[n] #  gt= 10083
        gt_channel=l_gt_channel[n] 
        l_recommened_list_user=l_recommened_list[n]
        html_user= html_user_info_with_feature (dataset_path,dic_map_users, dic_map_items, dic_map_channels, users_real, items_real,\
                 df_iter, user, gt, gt_channel, l_recommened_list_user )
        html_out+=html_user

    html_out+= "<p></p><h2>Top recomendations sample</h2>"
    NUMBERTOPSAMPLES=5
    nsample=NUMBERTOPSAMPLES
    for n in range (NUMBERFIRSTSAMPLES+1, num_users):        
        user = l_users[n] # user=3
        gt = l_gt_item[n] #  gt= 10083
        gt_channel=l_gt_channel[n] 
        l_recommened_list_user=l_recommened_list[n]
        if gt == l_recommened_list_user[0]: # gt == first recomendatiosn
            html_user= html_user_info_with_feature (dataset_path,dic_map_users, dic_map_items, dic_map_channels, users_real, items_real,\
                 df_iter, user, gt, gt_channel, l_recommened_list_user )
            html_out+=html_user
            nsample-=1
            if nsample==0:
                break
    html_out+= "<p></p><h2>Predictions without GT sample</h2>"
    NUMBERBADSAMPLES=5
    nsample=NUMBERBADSAMPLES
    for n in range (NUMBERFIRSTSAMPLES+1, num_users):        
        user = l_users[n] # user=3
        gt = l_gt_item[n] #  gt= 10083
        gt_channel=l_gt_channel[n] 
        l_recommened_list_user=l_recommened_list[n]
        if gt not in l_recommened_list_user: # gt == first recomendatiosn
            html_user= html_user_info_with_feature (dataset_path,dic_map_users, dic_map_items, dic_map_channels, users_real, items_real,\
                 df_iter, user, gt, gt_channel, l_recommened_list_user )
            html_out+=html_user
            nsample-=1
            if nsample==0:
                break

           

    auxs ="Report_{}.html".format(title.replace(" ",""))
    with open(auxs, 'w') as f:
        f.write(html_out)


   
    
