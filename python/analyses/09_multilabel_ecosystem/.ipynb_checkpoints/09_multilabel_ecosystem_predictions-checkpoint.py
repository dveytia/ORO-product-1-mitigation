import sys
print(sys.version)
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import ast
import time
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import tensorflow_addons as tfa
# Add these modules if eval=True
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score
import gc
import os

t0 = time.time()


################# Change INPUTS ##################
targetVar = "ecosystem" # name of variable
v='1' # Version of the script
dataFolder = '/homedata/dveytia/Product_1_data'

codedVariablesTxt1 = f'{dataFolder}/data/articleAnswers_formatted_2025-03-17.txt'
codedVariablesTxt2 = f'{dataFolder}/data/articleAnswers_notMRE_formatted_2025-05-26.txt' # file with coded variables

n_threads = 2 # number of threads to parallelize on
n_folds = 5
rank_j = rank%n_folds # get rank
modType= 'multi-label' # type of model functions to load: either 'multi-label' or 'binary-label'
cv_results_fp = f'{dataFolder}/outputs/predictions/{targetVar}_predictions_eval_v{v}_k{rank_j}.csv'
model_weights_fp = f'{dataFolder}/outputs/model_weights'

############################# Load data ###############################
######################## Change file paths x3 #########################
# Load seen documents
seen_df1 = pd.read_csv(codedVariablesTxt1, delimiter='\t') 
seen_df2 = pd.read_csv(codedVariablesTxt2, delimiter='\t') 
seen_df = pd.concat([seen_df1, seen_df2]).drop_duplicates(subset='id')
seen_df = seen_df.dropna(subset=["abstract"])
seen_df['seen']=1
# Keep only rows relevant for oro labels
oroCols = [x for x in seen_df.columns if 'oro_type' in x]
seen_df = seen_df[seen_df[oroCols].eq(1).any(axis=1)]


############# Load unseen data for predictions ##############
# Load unseen documents & apply prediction boundaries for mitigation branch

# Original map predictions
unseen_df = pd.read_csv(f'{dataFolder}/data/all_unseen_mitigation_oros.txt', delimiter='\t') 
unseen_df=unseen_df.dropna(subset=['abstract']).reset_index(drop=True)
# Choose which prediction boundaries to apply. 
unseen_df = unseen_df[(unseen_df['oro_any.M_Renewables - mean_prediction']>=0.5) | (unseen_df['oro_any.M_Increase_efficiency - mean_prediction']>=0.5) | (unseen_df['oro_any.M_CO2_removal_or_storage - mean_prediction']>=0.5)]
unseen_df['seen']=0

# 2025 update
unseen_df2 = pd.read_csv(f'{dataFolder}/data/unique_references_UPDATE_13-05-2025.txt', delimiter='\t')
unseen_df2 = unseen_df2.rename(columns={'analysis_id':'id'})
unseen_df2=unseen_df2.dropna(subset=['abstract']).reset_index(drop=True)
pred_screen = pd.read_csv(f'{dataFolder}/outputs/predictions-compiled/oro_screen_update_predictions.csv')
pred_branch = pd.read_csv(f'{dataFolder}/outputs/predictions-compiled/oro_branch_update_predictions.csv')
unseen_df2 = unseen_df2.merge(pred_screen, how="left").merge(pred_branch, how="left")
# Choose which predictiction boundaries to apply
unseen_df2 = unseen_df2[unseen_df2['0 - relevance - mean_prediction']>=0.5] 
unseen_df2 = unseen_df2[unseen_df2['oro_branch.Mitigation - mean_prediction']>=0.5]
unseen_df2['seen']=0

# Merge two unseen data frames together
unseen_df = (pd.concat([unseen_df[["id","title", "abstract", "keywords","seen"]], unseen_df2[["id","title", "abstract", "keywords","seen"]]])
             .sort_values('id')
             .sample(frac=1, random_state=1)
             .reset_index(drop=True)
            )



# Concatenate seen and unseen #######################
df = (pd.concat([seen_df,unseen_df])
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)


# Concatenate all text together
df["keywords"] = 'Keywords: '+ df["keywords"]
df["text"] = df[["title", "abstract", "keywords"]].agg(
    lambda x: ' '.join(x.dropna()), axis=1
)

seen_index = df[df['seen']==1].index
unseen_index = df[df['seen']==0].index

print("Dataset has been re-formatted and is ready")



################ Start defining functions ############################
tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(n_threads)

MODEL_NAME = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
   
with open(f'/home/dveytia/IPython_Notebooks/Product_1/pyFunctions/{modType}_1_predictions_functions.py') as f:
    exec(f.read())

outer_scores = []
clfs = []
    

##################### Select targets here ###########################
targets = [x for x in df.columns if targetVar in x] 
df['labels'] = list(df[targets].values)

class_weight = {}
try:
    for i, t in enumerate(targets):
        cw = df[(df['random_sample']==1) & (df[t]==0)].shape[0] / df[(df['random_sample']==1) & (df[t]==1)].shape[0]
        class_weight[i] = cw
except:
    class_weight=None

outer_scores = []
clfs = []


parallel=False

################### Load best model #####################
outer_scores = []
inner_scores = []
params = ['batch_size','weight_decay','learning_rate','num_epochs','class_weight']

for k in range(3): 
    inner_df = pd.read_csv(f'{dataFolder}/outputs/model_selection/{targetVar}_model_selection_v{v}_k{k}.csv') 
    inner_df = inner_df.sort_values('F1 macro',ascending=False).reset_index(drop=True)
    inner_scores += inner_df.to_dict('records')

inner_scores = pd.DataFrame.from_dict(inner_scores).fillna(-1)
inner_scores['F1 - tp'] = inner_scores.loc[:, [col for col in inner_scores.columns if col.startswith('F1 -') and any(target in col for target in targets)]].mean(axis=1)

best_model = (inner_scores
              .groupby(params)['F1 - tp']
              .mean()
              .sort_values(ascending=False)
              .reset_index() 
             ).to_dict('records')[0]

del best_model['F1 - tp']
print(best_model)
if best_model['class_weight']==-1:
    best_model['class_weight']=None
else:
    best_model['class_weight'] = ast.literal_eval(best_model['class_weight'])


######################### Run model #######################################
##################### Change paths x2 #####################################
# outer_cv = KFold(n_splits=n_folds)

## Use stratified K fold instead to try and ensure target classes are in test set
def myStratifiedKFoldKFold(n_splits, df, targets, shuffle=True):

    # Convert binary labels to a list of present classes
    df["label_set"] = df[targets].apply(lambda row: np.where(row == 1)[0].tolist(), axis=1)

    # Create a primary label for stratification (first label of each row, fallback to -1 if empty)
    df["primary_label"] = df["label_set"].apply(lambda x: x[0] if x else -1)

    # Keep unique indices with a single primary label
    unique_df = df[df["primary_label"] != -1].copy()

    # # Get an index of the seen articles
    unique_df_seen_index = unique_df[unique_df['seen']==1].index

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=42)

    for train_idx, test_idx in skf.split(unique_df_seen_index, unique_df["primary_label"]):
        train_ids = unique_df.index[train_idx]
        test_ids = unique_df.index[test_idx]

        # Get all rows in df that match the selected unique indices
        train_set = df[df.index.isin(train_ids)]
        test_set = df[df.index.isin(test_ids)]
        # Retrieve the corresponding original indices
        train = train_set.index
        test = test_set.index
        yield (train, test)  # Return original indices

outer_cv = myStratifiedKFoldKFold(n_folds, df, targets, shuffle=True)

## test my new function
"""
for i, (train, test) in enumerate(outer_cv):
    print(f"Fold {i}:")
    print(f"  Train: index={len(train)}")
    print(f"  Test:  index={len(test)}")  
    train = train
    test = unseen_index
"""


# for k, (train, test) in enumerate(outer_cv.split(seen_index)):  
for k, (train, test) in enumerate(outer_cv):
    if k!=rank_j:
        continue

    test = unseen_index

    y_preds, model = train_eval_save_bert(best_model, df=df, train=train, test=test, evaluate=False)

    # Save results
    mw_fp = f'{model_weights_fp}/{targetVar}_{n_folds}fold/v{v}/k{k}'
    if not os.path.exists(mw_fp):
        os.makedirs(mw_fp)
        print(f'Weights directory created: {mw_fp}')
    else:
        print(f'Weights directory exists: {mw_fp}')
    model.save_pretrained(mw_fp, from_pt=True) 
    # # note to load the model: 
    # model = TFDistilBertForSequenceClassification.from_pretrained(mw_fp)

    np.save(f'{dataFolder}/outputs/predictions/{targetVar}_y_preds_{n_folds}fold_data_v{v}_k{k}.npz',y_preds)
    gc.collect()

np.save(f'/homedata/dveytia/Product_1_data/outputs/predictions_data/{targetVar}_{n_folds}fold_data_pred_ids_v{v}.npz',df.loc[unseen_index,"id"]) #Change file path 

print(t0 - time.time())
