
import torch
import os
from model_Popularity import Popularity_Recommender
from train import testpartial , train_one_epoch
from build_dataset import CustomerArticleDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import pandas as pd
# 
from utils_report import info_model_report

class Popularity_Recommender_OLD(torch.nn.Module):

	# Initialize all the variables
	def __init__(self):
		super(Popularity_Recommender, self).__init__()
		# Tha training data which is been provided.
		self.train_data = None #interactions
		self.user_id = None #Column for customers or users
		self.item_id = None #Column for articles or items
		self.popularity_recommendations = None #Ranking

	# Create the recommendations.
	def create(self,train_data,user_id,item_id,label):

		self.train_data = train_data
		self.user_id = user_id
		self.item_id = item_id
		self.label = label
		
		# The items are grouped by item_id aggregated with the sum of 1 in the labels, we only count the real interactions.
		train_data_grouped = train_data.groupby([self.item_id]).agg({self.label: 'sum'}).reset_index() 
		train_data_grouped.rename(columns = {self.label : 'score'}, inplace = True)

		self.popularity_recommendations = train_data_grouped


	# Method to user created recommendations
	#def predict(self, user_id, topk=10):
	def predict(self, interactions, device=None):
		debug = False
		Ranking = self.popularity_recommendations
		InteractionsDF = pd.DataFrame(interactions,columns=[self.user_id, self.item_id])
		InterationsRanked = pd.merge( InteractionsDF, Ranking, on=self.item_id,  how='left')
		
		#list_recommendation = torch.FloatTensor(InterationsRanked[self.item_id])

		if debug:
				print("======RANKING=======")	
				print(Ranking.head())
				print("======INTERACTIONS=======")	
				print(InteractionsDF.head())
				print("======MERGE INTERACTIONS & RANK=======")	
				print(InterationsRanked.sort_values(by=['score'], ascending = False))

		return torch.FloatTensor(InterationsRanked['score'])




if __name__ == '__main__':

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logs_base_dir = "runHM_10000"
    dataset_path = "../data/"
    
    os.makedirs(logs_base_dir, exist_ok=True)

    tb_Popu = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_POPU/')
    
    full_dataset = CustomerArticleDataset(dataset_path, num_negatives_train=4, num_negatives_test=99)
    data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)

    #generar modelo
    #model = Popularity_Recommender()
    #model.create(pd.DataFrame(full_dataset.interactions, columns=["customer_id", 'article_id', 'label']), "customer_id", 'article_id', 'label')
    model= Popularity_Recommender(full_dataset.train_mat)
    

    
    #Train the model
    tb = True
    topk = 10
    num_epochs=2

    for epoch_i in range(num_epochs):
        #train_loss = train_one_epoch(model, optimizer, data_loader, criterion, device)
        train_loss=0

        
        hr, ndcg, cov, gini, dict_recomend, nov, l_info = testpartial(model, full_dataset, device, topk=topk)

        print(f'epoch {epoch_i}:')
        print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f}, COV@{topk} = {cov:.4f}, GINI@{topk} = {gini:.4f}, NOV@{topk} = {nov:.4f} ')

        if tb_Popu:
            tb_Popu.add_scalar('train/loss', train_loss, epoch_i)
            tb_Popu.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_Popu.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)
            tb_Popu.add_scalar(f'eval/COV@{topk}', cov, epoch_i)
            tb_Popu.add_scalar(f'eval/GINI@{topk}', gini, epoch_i)
            tb_Popu.add_scalar(f'eval/NOV@{topk}', nov, epoch_i)

	# Specify a path to save to
    PATH = "Popularity_Recommender.pt"
    torch.save(model.state_dict(), PATH)

    res_header=[f"HR@{topk}", f"NDCG@{topk}", f"COV@{topk}",f"GINI@{topk}",f"NOV@{topk}" ]
    res_values=[f"{hr:.4f}", f"{ndcg:.4f}", f"{cov:.4f}", f"{gini:.4f}", f"{nov:.4f}"  ]
    res_info=[res_header,res_values]
    dataset_path = "../data/"


    info_model_report (model, dataset_path, res_info, l_info, \
            full_dataset, dict_recomend, title="Popularity Recommender - Partial", topk=10 )
