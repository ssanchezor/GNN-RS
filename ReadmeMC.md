# Readme MC
## Table of Contents <a name="toc"></a>

[X .TOC](#1-intro)
 -[A. Environment Requirements](#A-requirements)
    - [A.1. Software](#A1-software)
    - [A.2. Hardware](#A2-hardware)

 -[B. Running the code](#B-running)
    - [B.1. Dataset creation](#B1-dataset)
    - [B.2. Model execution withour features](#B2-models-nofeat)
    
## A. Enviroment Requirements <a name='A-requirements'>

### A.1. Software  <a name='A1-software'></a>

All project has been based on python and related libraries.

In order to use a controlled environment we have used conda.

##### Miniconda

We assume python and conda are installed.
```
laika@casiopea:~$ python3 -V
Python 3.8.10
laika@casiopea:~$ conda -V
conda 4.13.0
```
Then we create the environment and active it.

```
conda create --name jupyter_geo python=3.8
conda activate jupyter_geo
```
Then we install all de required packages:

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg
conda install scikit-learn-intelex
conda install -c conda-forge tensorboard
conda install -c conda-forge gdown 
conda install -c anaconda jupyter 
conda install -c pytorch torchtext 
conda install -c anaconda more-itertools 
conda install -c conda-forge spacy 
conda install -c conda-forge matplotlib 
```

### A.2. Hardware  <a name='A2-hardware'></a> 

We have used Google Collab for many of the early stages of the software development. 

Once we have started more in deep work, we have used:

* Local machine -> Ubuntu - I7 machine 48 GB with 2 x 1070 RTX 8 GB GPUs

* Google Cloud Platform -> 4 GPU x Tesla T4.


## B. Running the code <a name='B-running'></a>

As the kaggle's original data for all the full dataset of H&M is to big, we have had to create a filtering program in order to reduce the dataset and create the input files for the model processing.

### B.1. Dataset creation <a name="B1-dataset"></a>

#### Starting point: Full Kaggle dataset

Full dataset has been download from kaggle's  ["H&M Personalized Fashion Recommendations"](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)

Files are:

|             File         | | 
|:------------------------:|:-------:|
|articles.csv     | Article information related (article id, product code, ...). This file will be used later for report generation|
|customers.csv                | Customer information related (article id, product code, ...). This file will be used later for report generation|
|transactions_train.csv   | Transaction file. This file has the main information we will use for the RS models (t_date(transaction date), customer_id, article_id, price, sales_channel_id |
|images   | Image directory of the aprox 106K items |
 
 #### Data filtering (data_filtering.py)
 
 We will assume Kaggle files will be in:
 ```
 ../data/articles.csv
 ../data/customers.csv
 ../data/transactions_train.csv
 ```
 
 The following restrictions can be set with this program:
 - As transaction data spawns from  catl `20-Sep-2018` till `22-Sep-2020`, we can select the temporal range. For our models we have always set the inital data `2019-09-22` (program will read only dates before this one.)
 - We want to take in account articles with a minimu number of transactions (we understand that it should be a minimun threshold of the article importance. In all our models we have requested that only items with at least `5` transactions should be considered. 
 - In order to work with customers, we need them with a minimum significance. We can set the minimum number a customer must have to be take in account. We have set at least `20` transactions per customer in order to be considered in all models.
 - At the end, we have set the size of the our dataset depending the number of users to be randomly selected once the previous restrictions are set. He have used an smaller dataset with `10000` customers and a much larger of `80000` customers. 
 
These restrictions are hardcode in the program. These lines need to be changed in the main procedure
```
ini_date= "2019-09-22" 
min_trans_per_article=5
min_trans_per_customer=20
total_customers=10000
```
The file will generate automaticallly a csv file with these convention:

```
../data/transactions_ddup_{ini_date}_nart_{min_art}_ncust_{min_trans}_ncustr_{customer_number}.csv
```
For example for the selected values, result file will be:

```
transactions_ddup_2019-09-22_nart_5_ncust_20_ncustr_10000.csv
```

This file is the one it will be used as data for the models.

Just to get an idea, these 2 settings will represent:

| Customers dataset | Number of items | Number of transactions|
|:-----------------:|:---------------:|:---------------------:|
|10.000| 38.200 | 401.000 |
|80.000| 52.600 | 3.272.000 |

#### Working files for models without features with `build_dataset.py`

The dataset we will be use for our models depend if there are some feature or not. First case is just without any feture.

So, we need to transform the previous filtered csv in to working files that will resemble the Movilens indexed format for all items and customers. The program that will make this conversion is `build_dataset.py`

We will need to specifiy the csv with our filtered dataset in the main procedure
```
if __name__ == '__main__':
    transactions = transactions = pd.read_csv("../data/transactions_ddup_2019-09-22_nart_5_ncust_20_ncustr_80000.csv")
```
After running the program, it will generate the working files we need to use to run the models

Customer-article transactions are sorted by users and then by date for each user. The first transaction will be used for the test dataset.

##### ../data/customer.test.article

This is the general format:
| customer_id | article_id | label| t_dat|
|:-----------:|:----------:|:----:|:----:|
|...|...|...|...|
|...|...|...|...|
|user i-1|item 1 |1|tdat 1|
|user i|item 1|1|tdat 1|
|user i+1|item 1 |1|tdat 1|
|...|...|...|...|

For example the first transaction of the first 4 users:
| customer_id | article_id | label| t_dat|
|:-----------:|:----------:|:----:|:----:|
|1	|1	|1	|20200519|
|2	|27	|1	|20200618|
|3	|63|	1	|20200613|
|4	|92|	1	|20200618|
|...|...|...|...|
|...|...|...|...|

##### ../data/customer.train.article

Train data will contain the remain transactions for each user:

| customer_id | article_id | label| t_dat|
|:-----------:|:----------:|:----:|:----:|
|...|...|...|...|
|...|...|...|...|
|user i-1|item m |1|tdat m|
|user i|item 2|1|tdat 2|
|user i|item 3||tdat 3|
|...|...|...|...|
|user i|item n |1|tdat n|
|user i+1|item 2 |1|tdat 2|
|...|...|...|...|
|...|...|...|...|

For instance for user 38:

| customer_id | article_id | label| t_dat|
|:-----------:|:----------:|:----:|:----:|
|...|...|...|...|
|...|...|...|...|
|37	|1241	|1	|20190927|
|38	|548	|1	|20200901|
|38	|1243	|1	|20200901|
|38	|1244	|1	|20200831|
|38	|1245	|1	|20200831|
|38	|551	|1	|20200831|
|38	|1246	|1	|20200831|
|...|...|...|...|
|38	|1279	|1	|20191123|
|38	|350	|1	|20191123|
|38	|1280	|1	|20191123|
|39	|1282	|1	|20200903|
|...|...|...|...|
|...|...|...|...|

##### ../data/customer.dic

This is dictionary file that will allow us to relate each indexed customer with the actual customer value in original H&M dataset. This will be use to generate our reports only.

| actual customer_id | indexed customer_id | 
|-------------------:|:------------------:|
|00011770272...3ccbb8d5dbb4b92	|1|
|00023e3dd86...e0038c6b10f655a	|2|
|0002d2ef78e...bde96e828d770f6	|3|
|0002e6cdaab...98119fa9cb3cee7	|4|
|00045027219...d9a84994678ac71	|5|
|...|...|
|...|...|

##### ../data/article.dic

This is dictionary file that will allow us to relate each indexed article with the actual article value in original H&M dataset. This will be use to generate our reports only.

| actual aticle_id | indexed article_id | 
|-------------------:|:------------------:|
|708485004|	1|
|820462009|	2|
|863456003|	3|
|570002090|	4|
|863456005|	5|
|...|...|
|...|...|


#### Working files for models with features with `build_dataset_features.py`

The dataset we will be use for our models depend if there are some feature or not. Second case is just without the sales channel value.

The program that will make this conversion is `build_dataset_features.py`

We will need to specifiy the csv with our filtered dataset in the main procedure
```
if __name__ == '__main__':
    transactions = transactions = pd.read_csv("../data/transactions_ddup_2019-09-22_nart_5_ncust_20_ncustr_80000.csv")
```
After running the program, it will generate the working files we need to use to run the models

Customer-article transactions are sorted by users and then by date for each user. The first transaction will be used for the test dataset.

##### ../data/customer.test.article.channel

This is the general format:
| customer_id | article_id | chanel_id |label| t_dat|
|:-----------:|:----------:|:----:|:----:|:----:|
|...|...|...|...|...|
|...|...|...|...|...|
|user i-1|item 1| channel 1 |1|tdat 1|
|user i|item 1|channel 1 |1|tdat 1|
|user i+1|item 1 |channel 1 |1|tdat 1|
|...|...|...|...|...|

For example the first transaction of the first 4 users:
| customer_id | article_id | label| t_dat|
|:-----------:|:----------:|:----:|:----:|
|1	|1	|1	|1	|20200519|
|2	|27	|2	|1	|20200618|
|3	|63|1	|	1	|20200613|
|4	|92|2	|	1	|20200618|
|...|...|...|...|
|...|...|...|...|

##### ../data/customer.train.article.channel

Train data will contain the remain transactions for each user:

| customer_id | article_id |  chanel_id | label| t_dat|
|:-----------:|:----------:|:----:|:----:|:----:|
|...|...|...|...|...|
|...|...|...|...|...|
|user i-1|item m |channel m |1|tdat m|
|user i|item 2|channel 2| 1|tdat 2|
|user i|item 3|channel 3||tdat 3|
|...|...|...|...|
|user i|item n |channel n|1|tdat n|
|user i+1|item 2 |channel 2|1|tdat 2|
|...|...|...|...|...|
|...|...|...|...|...|

For instance for user 38:

| customer_id | article_id | label| t_dat|
|:-----------:|:----------:|:----:|:----:|
|...|...|...|...|
|...|...|...|...|
|37	|1241	|2	|1	|20190927|
|38	|548	|2	|1	|20200901|
|38	|1243	|2	|1	|20200901|
|38	|1244	|2	|1	|20200831|
|...|...|...|...|
|38	|350	|1	|1	|20191123|
|38	|1280	|1	|1	|20191123|
|39	|1282	|1	|1	|20200903|
|...|...|...|...|...|
|...|...|...|...|...|

##### ../data/customer.dic

This is dictionary file that will allow us to relate each indexed customer with the actual customer value in original H&M dataset. Remains the same for featured dataset.

##### ../data/article.dic

This is dictionary file that will allow us to relate each indexed article with the actual article value in original H&M dataset. Remains the same for featured dataset.

### B.2. Model execution without features <a name="B2-models-nofeat"></a>

All model share a very common pattern

| Program | Description |
|:------------------------:|:-------:|
|model_MODEL_NAME.py| Definition of the model class |
|main_MODEL_NAME.py| Main program in order to create an instance of the the dataset, another instance of the loader, another instance of the model and the loss function and the optimizer, and finally the loop for each epoch in order to tranin the epoch and evaulate the the dataset |


#### B.2.1 Random model <a name="B2-models-nofeat-random"></a>

In 'model_Random.py' there is a simple class definition that will pick a random item

'main_Random.py' will 
* Create a instance of the dataset (`full_dataset = CustomerArticleDataset(...)`)
* Create a dataloder instance (`data_loader = DataLoader(...)`)
* Create model instance (`model=RandomModel(..)`) 

There are no epochs, so the only remaining task are:
* There is no training. The predictions will generated when we test the test dataset
* We will verifiy the accuraccy of the dataset:`metric1, metric2...= testpartial(model, full_dataset, device, topk=topk)`
* We will generate a report displaying actual H&M data

#### B.2.2 Popularity model <a name="B2-models-nofeat-poularity"></a>

In 'model_Popularity.py' there is a simple class definition that will pick the most popular items

'main_Popularity.py' will 
* Create a instance of the dataset (`full_dataset = CustomerArticleDataset(...)`)
* Create a dataloder instance (`data_loader = DataLoader(...)`)
* Create model instance (` model= Popularity_Recommender(full_dataset.train_mat)`) 

There are no epochs, so the only remaining task are:
* There is no training. The predictions will generated when we test the test dataset
* We will verifiy the accuraccy of the dataset:`metric1, metric2...= testpartial(model, full_dataset, device, topk=topk)`
* We will generate a report displaying actual H&M data

#### B.2.3 Factorization Machines model <a name="B2-models-nofeat-FM"></a>
