import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import shuffle


def remove_duplicates(transaction):
    transaction=transaction.drop_duplicates()
    print(f' \tDuplicate records removed!')
    return transaction


def filter_by_date(transaction, date):
    datefilter_mask=transaction["t_dat"]>=date # mask
    transaction=transaction[datefilter_mask] # apply mask
    print(f' \tTransactions filtered from {date}')
    return transaction


def filter_by_article(transaction, min_article):
    number_dif_art=transaction.article_id.value_counts() # counts number records for each article_id
    transaction=transaction[transaction.article_id.isin(number_dif_art.index[number_dif_art.gt(min_article-1)])]
    print(f' \tTransactions filtered by minimum {min_article} articles')
    return transaction


def filter_by_costumer(transaction, min_customer):
    number_dif_cust=transaction.customer_id.value_counts() # counts number records for each article_id
    transaction=transaction[transaction.customer_id.isin(number_dif_cust.index[number_dif_cust.gt(min_customer-1)])]
    print(f' \tTransactions filtered by minimum {min_customer} customers')
    return transaction


def reduce_dataset_by_cust(transaction, num_customers):
    customers_aux=transaction.customer_id.value_counts()
    customers_aux=shuffle(customers_aux)
    customers_aux=customers_aux[0:num_customers]
    transaction=transaction[transaction.customer_id.isin(customers_aux.index)]
    print(f' \tDataset reduced to {num_customers} customers')
    return transaction


articles = pd.read_csv("../data/articles.csv")
customers = pd.read_csv("../data/customers.csv")
transactions = pd.read_csv("../data/transactions_train.csv")
ini_date= "2019-09-22" 
min_trans_per_article=5
min_trans_per_customer=20
total_customers=10000

transactions_nodup=remove_duplicates(transactions)
transactions_fdate=filter_by_date(transactions_nodup, ini_date)
transactions_fdate_fart=filter_by_article(transactions_fdate, min_trans_per_article)
transactions_fdate_fart_fcust=filter_by_costumer(transactions_fdate_fart, min_trans_per_customer)
transactions_fdate_fart_fcust_reduced=reduce_dataset_by_cust(transactions_fdate_fart_fcust, total_customers)

file_name="../data/transactions_ddup_{0}_nart_{1}_ncust_{2}.csv".format( \
   ini_date, min_trans_per_article, min_trans_per_customer)
file_name_red="../data/transactions_ddup_{0}_nart_{1}_ncust_{2}_ncustr_{3}.csv".format( \
   ini_date, min_trans_per_article, min_trans_per_customer, total_customers)

print(f' \tFull filtered dataset will be exported to {file_name} path')
transactions_fdate_fart_fcust.to_csv(file_name)

print(f' \tReduced filtered dataset will be exported to {file_name_red} path')
transactions_fdate_fart_fcust_reduced.to_csv(file_name_red)

print("Done!")