import pandas as pd
from sklearn.utils import shuffle


def remove_duplicates(transaction):
    # removes possible transaction duplicates
    transaction=transaction.drop_duplicates() # drops duplicates
    print(f' \tDuplicate records removed!')
    return transaction


def filter_by_date(transaction, date):
    # filters transactions on a given date
    datefilter_mask=transaction["t_dat"]>=date # extracts date mask
    transaction=transaction[datefilter_mask] # applies date mask to filter transactions
    print(f' \tTransactions filtered from {date}')
    return transaction


def filter_by_article(transaction, min_article):
    # filters transactions on a given number of minimum purchases per article
    number_dif_art=transaction.article_id.value_counts() # counts number records for each article_id
    transaction=transaction[transaction.article_id.isin(number_dif_art.index[number_dif_art.gt(min_article-1)])] # drops articles below the minimum
    print(f' \tTransactions filtered by minimum {min_article} articles')
    return transaction


def filter_by_costumer(transaction, min_customer):
    # filters transactions on a given number of minimum purchases per customer
    number_dif_cust=transaction.customer_id.value_counts() # counts number records for each customer_id
    transaction=transaction[transaction.customer_id.isin(number_dif_cust.index[number_dif_cust.gt(min_customer-1)])] # drops customers below the minimum
    print(f' \tTransactions filtered by minimum {min_customer} customers')
    return transaction


def reduce_dataset_by_cust(transaction, num_customers):
    # reduces dataset on a given number of customers
    customers_aux=transaction.customer_id.value_counts() # counts number of costumers
    customers_aux=shuffle(customers_aux) # shuffles customers since they are ordered
    customers_aux=customers_aux[0:num_customers] # extracts customer mask
    transaction=transaction[transaction.customer_id.isin(customers_aux.index)] # applies customer mask to filter transactions
    print(f' \tDataset reduced to {num_customers} customers')
    return transaction

# preparing data...
articles = pd.read_csv("../data/articles.csv") # reads article data
customers = pd.read_csv("../data/customers.csv") # reads customer data
transactions = pd.read_csv("../data/transactions_train.csv") # reads transcations data
ini_date= "2019-09-22"  # we limit transactions for the last year
min_trans_per_article=5 # we limit articles that have been purchased more than 5 times
min_trans_per_customer=20 # we limit customers that have been bought more than 20 articles
total_customers=10000 # for the above conditions, we only extract a 10K / 80K customers.

# applying data filtering functions...
transactions_nodup=remove_duplicates(transactions)
transactions_fdate=filter_by_date(transactions_nodup, ini_date)
transactions_fdate_fart=filter_by_article(transactions_fdate, min_trans_per_article)
transactions_fdate_fart_fcust=filter_by_costumer(transactions_fdate_fart, min_trans_per_customer)
transactions_fdate_fart_fcust_reduced=reduce_dataset_by_cust(transactions_fdate_fart_fcust, total_customers)

# saving results...
file_name="../data/transactions_ddup_{0}_nart_{1}_ncust_{2}.csv".format( \
   ini_date, min_trans_per_article, min_trans_per_customer)
file_name_red="../data/transactions_ddup_{0}_nart_{1}_ncust_{2}_ncustr_{3}.csv".format( \
   ini_date, min_trans_per_article, min_trans_per_customer, total_customers)
print(f' \tFull filtered dataset will be exported to {file_name} path')
transactions_fdate_fart_fcust.to_csv(file_name)
print(f' \tReduced filtered dataset will be exported to {file_name_red} path')
transactions_fdate_fart_fcust_reduced.to_csv(file_name_red)

print("Done!")