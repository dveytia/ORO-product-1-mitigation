import sys
print(sys.version)
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score
import itertools
import time

t0 = time.time()


################# Change INPUTS ##################
targetVar = "oro_type.MRE" # name of variable
v='2_MreLO' # Version of the script
dataFolder = '/homedata/dveytia/Product_1_data'
codedVariablesTxt = f'{dataFolder}/data/articleAnswers_formatted_2025-03-17.txt' # file with coded variables
n_threads = 3 # number of threads to parallelize on
n_folds = 3 # 3
rank_j = rank%n_folds # get rank
modType= 'multi-label' # type of model functions to load: either 'multi-label' or 'binary-label'
cv_results_fp = f'{dataFolder}/outputs/model_selection/{targetVar}_model_selection_v{v}_k{rank_j}.csv'

############################# Load data ###############################
######################## Change file paths x2 #########################
df = pd.read_csv(codedVariablesTxt, delimiter='\t') 

# Keep screening decisions in
# df = df.loc[df["relevant"] == 1,]
df = df.dropna(subset=["abstract"]) # if any abstracts are NA, remove

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

######################### Define functions #############################

tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(n_threads)

with open(f'/home/dveytia/IPython_Notebooks/SPATMAN_map/pyFunctions/{modType}_0_model-selection_functions.py') as f:
    exec(f.read())

MODEL_NAME = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


###################### Select targets here #################################
targets = [x for x in df.columns if targetVar in x] 
targets = [x for x in targets if 'MRE-Bio' not in x] # But remove MRE-Bio

# Should I add an 'other' option? Right now no


# # if the target is not present in the random_sample, remove from targets?
# colSums = df.loc[df['random_sample']==1,targets].sum()
# targets = list(colSums[colSums > 0].index)

df['labels'] = list(df[targets].values)

class_weight = {}
for i, t in enumerate(targets):
    try:
        cw = df[(df['random_sample']==1) & (df[t]==0)].shape[0] / df[(df['random_sample']==1) & (df[t]==1)].shape[0]
    except:
        cw=0
    class_weight[i] = cw
    
class_weight

bert_params = {
  "class_weight": [None,class_weight],
  "batch_size": [16, 32],
  "weight_decay": (0, 0.3),
  "learning_rate": (1e-5, 5e-5),
  "num_epochs": [2, 3, 4]
}


param_space = list(product_dict(**bert_params))



## Use stratified K fold instead to try and ensure target classes are in test set
# outer_cv = KFoldRandom(n_folds, df.index, df['labels'],df[df['random_sample']!=1].index, discard=False)
def StratifiedKFoldKFoldRandom(n_splits, df, targets, no_test, shuffle=False, discard=True):

    # Convert binary labels to a list of present classes
    df["label_set"] = df[targets].apply(lambda row: np.where(row == 1)[0].tolist(), axis=1)

    # Create a primary label for stratification (first label of each row, fallback to -1 if empty)
    df["primary_label"] = df["label_set"].apply(lambda x: x[0] if x else -1)

    # Keep unique indices with a single primary label
    unique_df = df[df["primary_label"] != -1].copy()

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(unique_df.index, unique_df["primary_label"]):
        train_ids = unique_df.index[train_idx]
        test_ids = unique_df.index[test_idx]

        # Get all rows in df that match the selected unique indices
        train_set = df[df.index.isin(train_ids)]
        test_set = df[df.index.isin(test_ids)]
        # Retrieve the corresponding original indices
        train = train_set.index
        test = test_set.index
        
        # remove no_test values from the test set
        if not discard:
            train = list(train) +  [x for x in test if x in no_test]
        test = [x for x in test if x not in no_test]

        yield (train, test)  # Return original indices


outer_cv = StratifiedKFoldKFoldRandom(n_folds, df, targets, df[df['random_sample']!=1].index, discard=False)

## test my new function
# for i, (train_index, test_index) in enumerate(outer_cv):
#     print(f"Fold {i}:")
#     print(f"  Train: index={len(train_index)}")
#     print(f"  Test:  index={len(test_index)}")        


# outer_scores = []
# clfs = []




############################## Run models ################################
######################## Change file path (x3) ###########################
for k, (train, test) in enumerate(outer_cv):    
    if k!=rank_j:
        continue
    try:
        pr = param_space[0]
        cv_results=pd.read_csv(cv_results_fp).to_dict('records') 
        params_tested=pd.read_csv(cv_results_fp)[list(pr.keys())].to_dict('records')
    except:
        cv_results = []
        params_tested = []
    for pr in param_space:
        if pr in params_tested:
            continue
        cv_results.append(train_eval_bert(pr, df=df, train=train, test=test))
        pd.DataFrame.from_dict(cv_results).to_csv(cv_results_fp,index=False) 
        gc.collect()
