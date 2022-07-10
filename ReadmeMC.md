# Readme MC
## Table of Contents <a name="toc"></a>

- [X .TOC](#1-intro)
 -[A. Environment Requirements](#A-requirements)
    - [A.1. Software](#A1-software)
    - [A.2. Hardware](#A2-hardware)

 -[B. Running the code](#B-running)
    - [B.1. Dataset creation](#B1-dataset)
    - [B.2. Models execution](#B2-models)

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

##A.2. Hardware  <a name='A2-hardware'></a> 

We have used Google Collab for many of the early stages of the software development. 

Once we have started more in deep work, we have used:

* Local machine -> Ubuntu - I7 machine 48 GB with 2 x 1070 RTX 8 GB GPUs

* Google Cloud Platform -> 4 GPU x Tesla T4.


## B. Running the code <a name='B-running'></a>

As the kaggle's original data for all the full dataset of H&M is to big, we have had to create a filtering program in order to reduce the dataset and create the input files for the model processing.

### B.1. Dataset creation <a name="B1-dataset"></a>

#### Full dataset

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
|10.000| 38200 | 401.000 |
|80.000| 52.600 | 3.272.000 |
