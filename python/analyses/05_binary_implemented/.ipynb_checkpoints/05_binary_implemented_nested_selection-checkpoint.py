import sys
print(sys.version)
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()


import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score
import itertools
import time

t0 = time.time()


"""
This version explores the possibility/value of nesting predictions by ORO type. It turns out it would be needed to generate more believable predictions, but not enough data to train the model
"""

################# Change INPUTS ##################
targetVar = 'implemented' # name of variable
v='1' # Version of the script
dataFolder = '/homedata/dveytia/Product_1_data'

codedVariablesTxt1 = f'{dataFolder}/data/articleAnswers_formatted_2025-03-17.txt'
codedVariablesTxt2 = f'{dataFolder}/data/articleAnswers_notMRE_formatted_2025-05-26.txt' # file with coded variables

n_threads = 5 # number of threads to parallelize on
n_folds = 3
rank_j = rank%n_folds
modType= 'binary-label' # type of model functions to load: either 'multi-label' or 'binary-label'
cv_results_fp = f'{dataFolder}/outputs/model_selection/{targetVar}_model_selection_v{v}_k{rank_j}.csv'



############################# Load data ###############################
######################## Change file paths x2 #########################
df1 = pd.read_csv(codedVariablesTxt1, delimiter='\t') 
df1 = df1.dropna(subset=["abstract"])

df2 = pd.read_csv(codedVariablesTxt2, delimiter='\t') 
df2 = df2.dropna(subset=["abstract"])

df = pd.concat([df1, df2]).drop_duplicates(subset='id')

# Keep only rows relevant for oro labels
oroCols = [x for x in df.columns if 'oro_type' in x]
df = df[df[oroCols].eq(1).any(axis=1)]


# Sort values
df = (df
      #.query('unlabelled==0')
      # .query('relevant==1')
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)


# Concatenate all text together
df["keywords"] = 'Keywords: '+ df["keywords"]
df["text"] = df[["title", "abstract", "keywords"]].agg(
    lambda x: ' '.join(x.dropna()), axis=1
)

print("The data has been re-formatted")
print(df.shape)


## explore how many implemented for the different oro types
df.loc[df[targetVar]==1,[x for x in df.columns if 'oro_type' in x]].sum().sort_values(ascending=False)
"""
oro_type.CDR-BC         96
oro_type.MRE-Located    39
oro_type.MRE-Ocean      19
oro_type.Efficiency     19
oro_type.CDR-Cult        9
oro_type.CCS             7
oro_type.CDR-BioPump     5
oro_type.MRE-Bio         2
oro_type.CDR-OAE         1
oro_type.CDR-Other       1

This is odd because for predicted, only 35 implemented for MRE-Located but almost 1000 for MRE ocean... (see below) and nesting by type wouldn't work because the sample size isnt enough to train the model.

oro_type.MRE-Located - mean_prediction     35
oro_type.MRE-Ocean - mean_prediction      947
oro_type.CCS - mean_prediction              0
oro_type.CDR-BC - mean_prediction         918
oro_type.MRE-Bio - mean_prediction          0
oro_type.Efficiency - mean_prediction       0
oro_type.CDR-OAE - mean_prediction          0
oro_type.CDR-BioPump - mean_prediction      0
"""


