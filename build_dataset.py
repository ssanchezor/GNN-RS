import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from train import test


def add_label_column(transaction, value):
    transaction["label"] = value  # add label column with fixed value
    return transaction


def extract_dictionary(transaction):
    cust_dict = {};
    art_dict = {}
    count_cust = 1;
    count_art = 1;
    for index, row in transaction.iterrows():
        customer=row["customer_id"]; article=row["article_id"]
        if (customer not in cust_dict):
            cust_dict[customer]=count_cust; count_cust+=1
        if (article not in art_dict):
            art_dict[article]=count_art; count_art+=1
    return cust_dict, art_dict


def generate_datasets(transaction, cust_dict, art_dict):
    test_data_list = [];
    train_data_list = [];
    last_customer_id = -999;
    current_customer_id = -999
    for index, row in transaction.iterrows():
        customer = row["customer_id"];
        customer_id = cust_dict[customer]
        article = row["article_id"];
        article_id = art_dict[article]
        timestamp = int(row["t_dat"].replace('-', ''))
        if (last_customer_id != customer_id):
            current_customer_id = customer_id
            last_customer_id = customer_id
            row = [current_customer_id, article_id, row["label"], timestamp]
            test_data_list.append(row)
        else:
             row = [current_customer_id, article_id, row["label"], timestamp]
             train_data_list.append(row)
    len_test=len(test_data_list)
    len_train=len(train_data_list)
    if __name__ == "build_dataset":
        print(f' \tTest dataset generated, length:: {len_test}')
        print(f' \tTrain dataset generated, length: {len_train}')
    return test_data_list, train_data_list


def build_adj_mx(dims, interactions):
    adj_mat = sp.dok_matrix((dims, dims), dtype=np.float32)
    for x in tqdm(interactions, desc="BUILDING ADJACENCY MATRIX..."):
        adj_mat[x[0], x[1]] = 1.0
        adj_mat[x[1], x[0]] = 1.0
    return adj_mat


class CustomerArticleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, num_negatives_train=4, num_negatives_test=100, sep='\t'): 
        
        column_names = ["customer_id", "article_id", "label", "t_dat"]
        # MOVIE
        test_file_name= "movielens.test.rating"
        train_file_name ="movielens.train.rating"
        # H& H
        test_file_name= "customer.test.article"
        train_file_name ="customer.train.article"

        
        
        
        
        train_data = pd.read_csv(f'{dataset_path}{train_file_name}', sep=sep,
                                 header=None, names=column_names)
        number_customers= train_data["customer_id"].nunique() # mcanals first find number of customers
        train_data= train_data.to_numpy() # then convert to numpy 
        
        test_data = pd.read_csv(f'{dataset_path}{test_file_name}', sep=sep,
                                header=None, names=column_names)
        number_customers_test= test_data["customer_id"].nunique() # same for test
        test_data= test_data.to_numpy()
        
        assert  number_customers == number_customers_test, \
            f"number_customers in train {number_customers} should match in test {number_customers_test}"
            
        # TAKE items, targets and test_items
        self.targets = train_data[:, 2] # all array([1,...1]) for traindata
        self.items = self.preprocess_items(train_data, number_customers) # mcanals comment [cust1 , itemi], [cust1, itemk]...[cust_n, itemx]     
        
        # Save dimensions of max users and items and build training matrix
        self.field_dims = np.max(self.items, axis=0) + 1  # ([ ncustomers, totalilines])
        self.train_mat = build_adj_mx(self.field_dims[-1], self.items.copy()) #mcanals self.field_dims[-1] last value in list -> num articles

        # Generate train interactions with 4 negative samples for each positive in test data
        self.negative_sampling(num_negatives=num_negatives_train) # 

        # Build test set by passing as input the test item interactions
        # we keep the side but for evaluation we add 99 negativess for each sample
        # for each customer, 1st value is positive, 99 negatives # mcanals comment
        # negatives samples as 100 increases memory need in inference fyi # mcanals comment
        self.test_set = self.build_test_set(self.preprocess_items(test_data, number_customers),
                                            num_neg_samples_test=num_negatives_test)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.interactions[index]

    def preprocess_items(self, data, num_customers):
        reindexed_items = data[:, :2].astype(np.int) - 1 # -1 because ID begins from 1
        reindexed_items[:, 1] = reindexed_items[:, 1] + num_customers # no se si va -1 laika
        return reindexed_items

    def negative_sampling(self, num_negatives):
        self.interactions = []
        data = np.c_[(self.items, self.targets)].astype(int)
        max_users, max_items = self.field_dims[:2]  # number users (943), number items (2625)

        for x in tqdm(data, desc="Performing negative sampling on test data..."):  # x are triplets (u, i , 1)
            # Append positive interaction
            self.interactions.append(x)
            # Copy user and maintain last position to 0. Now we will need to update neg_triplet[1] with j
            neg_triplet = np.vstack([x, ] * (num_negatives))
            neg_triplet[:, 2] = np.zeros(num_negatives)

            # Generate num_negatives negative interactions
            for idx in range(num_negatives):
                j = np.random.randint(max_users, max_items)
                # IDEA: Loop to exclude true interactions (set to 1 in adj_train) user - item
                while (x[0], j) in self.train_mat:
                    j = np.random.randint(max_users, max_items)
                neg_triplet[:, 1][idx] = j
            self.interactions.append(neg_triplet.copy())

        self.interactions = np.vstack(self.interactions)

    def build_test_set(self, gt_test_interactions, num_neg_samples_test):
        max_users, max_items = self.field_dims[:2]  # number users (943), number items (2625)
        test_set = []
        for pair in tqdm(gt_test_interactions, desc="BUILDING TEST SET..."):
            negatives = []
            for t in range(num_neg_samples_test):
                j = np.random.randint(max_users, max_items)
                while (pair[0], j) in self.train_mat or j == pair[1]:
                    j = np.random.randint(max_users, max_items)
                negatives.append(j)
            # APPEND TEST SETS FOR SINGLE USER
            single_user_test_set = np.vstack([pair, ] * (len(negatives) + 1))
            single_user_test_set[:, 1][1:] = negatives
            test_set.append(single_user_test_set.copy())
        return test_set


if __name__ == '__main__':    
    transactions = pd.read_csv("../data/transactions_ddup_2019-09-22_nart_5_ncust_20_ncustr_10000.csv")
    
    transactions = add_label_column(transactions, 1)
    transactions = transactions.sort_values(['customer_id', 't_dat'], ascending=[True, False])

    customer_dict, article_dict = extract_dictionary(transactions)
    test_dataset, train_dataset = generate_datasets(transactions, customer_dict, article_dict)

    column_names = ["customer_id", "article_id", "label", "t_dat"]
    train_data = pd.DataFrame(train_dataset, columns= column_names)
    test_data = pd.DataFrame(test_dataset, columns= column_names)
    train_data.to_csv("../data/customer.train.article" , sep="\t", index=False,header=False)
    test_data.to_csv("../data/customer.test.article" , sep="\t", index=False,header=False)