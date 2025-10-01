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
import datasets
from datasets import Dataset, DatasetDict
import umap.umap_ as umap 
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
import pickle

t0 = time.time()


################# Change INPUTS ##################
targetVar = "oro_type" # name of variable
v='2_geoEng_UMAP' # Version of the script
dataFolder = '/homedata/dveytia/Product_1_data'
codedVariablesTxt = f'{dataFolder}/data/articleAnswers_notMRE_formatted_2025-05-26.txt' # file with coded variables
n_threads = 1 # number of threads to parallelize on
n_folds = 3
rank_j = rank%n_folds # get rank
modType= 'multi-label' # type of model functions to load: either 'multi-label' or 'binary-label'
cv_results_fp = f'{dataFolder}/outputs/model_selection/{targetVar}_model_selection_v{v}_k{rank_j}.csv'
model_weights_fp = f'{dataFolder}/outputs/model_weights'

############################# Load seen data ###############################
######################## Change file paths x2 #########################
df = pd.read_csv(codedVariablesTxt, delimiter='\t') 
df = df.dropna(subset=["abstract"]) # if any abstracts are NA, remove

# Remove articles relevant for MRE-Ocean or MRE-Located,
# Because predicting over nonMRE articles so in order for 'random' to be represenative,
# these need to be removed
df = df[df[f'{targetVar}.MRE-Located']==0]
df = df[df[f'{targetVar}.MRE-Ocean']==0]

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


############# Load unseen data for predictions ##############
# Load unseen documents & apply prediction boundaries
unseen_df = pd.read_csv(f'{dataFolder}/data/all_unseen_mitigation_oros.txt', delimiter='\t') 
#unseen_df = unseen_df.rename(columns={'analysis_id':'id'})
unseen_df=unseen_df.dropna(subset=['abstract']).reset_index(drop=True)
# Choose which prediction boundaries to apply. 
unseen_df = unseen_df[(unseen_df['oro_any.M_Renewables - mean_prediction']>=0.5) | (unseen_df['oro_any.M_Increase_efficiency - mean_prediction']>=0.5) | (unseen_df['oro_any.M_CO2_removal_or_storage - mean_prediction']>=0.5)]
unseen_df['seen']=0

# Load unseen updated documents & apply prediction boundaries to just Mitigation branch
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

# Concatenate all text together
unseen_df["keywords"] = 'Keywords: '+ unseen_df["keywords"]
unseen_df["text"] = unseen_df[["title", "abstract", "keywords"]].agg(
    lambda x: ' '.join(x.dropna()), axis=1
)

## Choose prediction bounndaries -- NOT MRE
pred_MRE = pd.read_csv(f'{dataFolder}/outputs/predictions-compiled/oro_type.MRE_v2_MreLO_predictions.csv')
unseen_df = unseen_df.merge(pred_MRE, how = "left")
unseen_df = unseen_df[(unseen_df['oro_type.MRE-Located - mean_prediction']<0.5) | (unseen_df['oro_type.MRE-Ocean - mean_prediction']<0.5)]


######################### Define functions #############################

tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(n_threads)

with open(f'/home/dveytia/IPython_Notebooks/Product_1/pyFunctions/{modType}_0_model-selection_functions.py') as f:
    exec(f.read())

MODEL_NAME = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


###################### Select targets here #################################
# Drop columns I don't want
dropColumns = ['oro_type.MRE-Located', 'oro_type.CCS', 'oro_type.MRE-Ocean', 'oro_type.CDR-BC', 'oro_type.MRE-Bio', 'oro_type.CDR-Other','oro_type.Efficiency']
df = df.drop(dropColumns, axis=1)

# Then get targets
targets = [x for x in df.columns if targetVar in x] 

# # Should I add an 'other' option? 
## Yes if not subsetting to just 'relevant' articles, because this acts as the 'screener;' 
## otherwise, predicts relevant to everything
df[f'{targetVar}-Absent'] = df[targets].eq(0).all(axis=1).astype(int)
targets.append(f'{targetVar}-Absent')



# Check how many samples in random sample per target
colSums = df.loc[df['random_sample']==1,targets].sum()
print(f"Column Sums: \n{colSums}")


df['labels'] = list(df[targets].values)


# Use targets to initialize model
n_labels = len(targets)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_labels)




########### Additional functions ####################

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

# ## test my new function
# outer_cv = StratifiedKFoldKFoldRandom(n_folds, df, targets, df[df['random_sample']!=1].index, discard=False)

# for i, (train, test) in enumerate(outer_cv):
#     if i != rank_j:
#         continue
#     print(f"Fold {i}:\n")
#     print(f"Training Sums:\n")
#     print(df.loc[train,targets].sum())
#     print(f"Testing Sums:\n")
#     print(df.loc[test,targets].sum())   
    




def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)



def get_output_embeddings(batch):
    inputs = {key: tf.convert_to_tensor(tensor) for key, tensor in batch.items() if key in ['input_ids', 'attention_mask']}
    output = model.distilbert(**inputs, training=False).last_hidden_state
    features = output[:, 0, :]
    return {"features": features.numpy()}


# process results for umap
# Function to transform labels
def transform_labels(example):
    # Replace 1s with corresponding labels, remove 0s
    example['labels'] = [i for i, val in enumerate(example['labels']) if val == 1]
    return example

def unnestDataset(dat):
    # Apply transformation
    tmpDat = dat.map(transform_labels)
    
    # Unnest dataset so each row has a single label
    unnested_data = {'id': [], 'labels': [], 'features': []}
    for row in tmpDat:
        for label in row['labels']:
            unnested_data['id'].append(row['id'])
            unnested_data['labels'].append(label)
            unnested_data['features'].append(row['features'])
    
    # Create final unnested Dataset
    unnested_dataset = Dataset.from_dict(unnested_data)
    return unnested_dataset

# def evaluate_preds(y_true, y_pred):
#     try:
#         roc_auc = roc_auc_score(y_true, y_pred)
#     except:
#         roc_auc = np.NaN
#     f1 = f1_score(y_true, y_pred.round())
#     p, r = precision_score(y_true, y_pred.round()), recall_score(y_true, y_pred.round())
#     acc = accuracy_score(y_true, y_pred.round())
#     print(f"ROC AUC: {roc_auc:.0%}, F1: {f1:.1%}, precision: {p:.1%}, recall {r:.1%}, acc {acc:.0%}")
#     return {"ROC AUC": roc_auc, "F1": f1, "precision": p, "recall": r, "accuracy": acc}


def evaluate_preds(y_true, y_pred, ids, targets, model_name):
    """Evaluates multi-label classification predictions, handling document-level aggregation."""
    warnings.filterwarnings("ignore")  # Suppress all warnings
    res = {}
    res['Model'] = model_name

    # Convert numerical labels to one-hot encoded format
    y_true_one_hot = np.zeros((len(y_true), len(targets)))
    y_pred_one_hot = np.zeros((len(y_pred), len(targets)))

    for i, label in enumerate(y_true):
        y_true_one_hot[i, label] = 1  # Set index position for label
    for i, label in enumerate(y_pred):
        y_pred_one_hot[i, label] = 1

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({"id": ids})
    y_true_df = pd.DataFrame(y_true_one_hot, columns=targets)
    y_pred_df = pd.DataFrame(y_pred_one_hot, columns=targets)

    # Aggregate at document level using max (assuming presence of a label in any row means it applies to doc)
    y_true_agg = y_true_df.groupby(df["id"]).max()
    y_pred_agg = y_pred_df.groupby(df["id"]).max()

    # Convert back to numpy
    y_true_agg = y_true_agg.values
    y_pred_agg = y_pred_agg.values

    # Compute evaluation metrics
    for average in ["micro", "macro", "weighted", "samples"]:
        try:
            res[f'ROC AUC {average}'] = roc_auc_score(y_true_agg, y_pred_agg, average=average)
        except:
            res[f'ROC AUC {average}'] = np.NaN
        res[f'F1 {average}'] = f1_score(y_true_agg, y_pred_agg.round(), average=average)
        res[f'precision {average}'] = precision_score(y_true_agg, y_pred_agg.round(), average=average)
        res[f'recall {average}'] = recall_score(y_true_agg, y_pred_agg.round(), average=average)

    print(f"{model_name} F1 macro: {res['F1 macro']}")

    # Per-label metrics
    for i, target in enumerate(targets):
        try:
            res[f'ROC AUC - {target}'] = roc_auc_score(y_true_agg[:, i], y_pred_agg[:, i])
        except:
            res[f'ROC AUC - {target}'] = np.NaN
        res[f'precision - {target}'] = precision_score(y_true_agg[:, i], y_pred_agg[:, i].round())
        res[f'recall - {target}'] = recall_score(y_true_agg[:, i], y_pred_agg[:, i].round())
        res[f'F1 - {target}'] = f1_score(y_true_agg[:, i], y_pred_agg[:, i].round())
        res[f'accuracy - {target}'] = accuracy_score(y_true_agg[:, i], y_pred_agg[:, i].round())
        res[f'n_target - {target}'] = y_true_agg[:, i].sum()

    return res






############################## Run models ################################
######################## Change file path (x3) ###########################

# ## For testing:
# tmp = df.groupby(targets).sample(n=1, random_state=1)
# train = tmp.index
# test = tmp.index

outer_cv = StratifiedKFoldKFoldRandom(n_folds, df, targets, df[df['random_sample']!=1].index, discard=False)

for k, (train, test) in enumerate(outer_cv):    
    if k!=rank_j:
        continue

    alldat = datasets.DatasetDict({
        "train": Dataset.from_pandas(df.loc[train,['text', 'id', 'labels']]),
        "test": Dataset.from_pandas(df.loc[test, ['text', 'id', 'labels']]),
        "predict": Dataset.from_pandas(unseen_df[['text', 'id']])
    })
    alldat_tokenized = alldat.map(tokenize_text, batched=True, batch_size=None)
    alldat_feat = alldat_tokenized.map(get_output_embeddings, batched=True, batch_size=1)

    # # Save extracted features?
    # with open(f'{dataFolder}/data/derived-data/{targetVar}_v{v}_k{k}_extractedFeatures.pkl', 'wb') as f:
    #     pickle.dump(alldat_feat, f)

    # Un-nest data and convert labels to numeric
    alldat_feat["train"] = unnestDataset(alldat_feat["train"])
    alldat_feat["test"] = unnestDataset(alldat_feat["test"])
    
    
    # Assign variables
    X_train = np.array(alldat_feat["train"]["features"])
    y_train = np.array(alldat_feat["train"]["labels"])
    X_test = np.array(alldat_feat["test"]["features"])
    y_test = np.array(alldat_feat["test"]["labels"])
    X_predict = np.array(alldat_feat["predict"]["features"])

    # Umap embeddings
    ndim = len(targets)-1
    mapper = umap.UMAP(min_dist=0,n_components=ndim).fit(X_train, np.array(y_train)) 
    umap_train = mapper.embedding_.T
    umap_test = mapper.transform(X_test).T 
    umap_predict = mapper.transform(X_predict).T

    # Classifier on embeddings
    rftmp = RandomForestClassifier()
    rftmp.fit(np.transpose(umap_train), y_train)
    preds = rftmp.predict(np.transpose(umap_test))

    # Save umap reducer and RF classifier
    mw_fp = f'{model_weights_fp}/{targetVar}_{n_folds}fold/v{v}/k{k}'
    if not os.path.exists(mw_fp):
        os.makedirs(mw_fp)
        print(f'Weights directory created: {mw_fp}')
    else:
        print(f'Weights directory exists: {mw_fp}')

    pickle.dump(mapper, open(f'{mw_fp}/umap_mapper.sav', 'wb'))
    pickle.dump(rftmp, open(f'{mw_fp}/rftmp.sav', 'wb'))
    # to reload:: mapper = pickle.load((open(f'{mw_fp}/umap_mapper.sav', 'rb')))
    # to reload:: rftmp = pickle.load((open(f'{mw_fp}/rftmp.sav', 'rb')))


    # Evaluate results & save
    eps = evaluate_preds(y_test, preds, alldat_feat["test"]["id"], targets, f"Random Forest on UMAP Dim {ndim}")
    eps["fold"] = k
    pd.DataFrame.from_dict([eps]).to_csv(cv_results_fp,index=False)

    # Predictions & save
    y_preds = rftmp.predict(np.transpose(umap_predict))
    np.save(f'{dataFolder}/outputs/predictions/{targetVar}_y_preds_{n_folds}fold_data_v{v}_k{k}.npz',y_preds)
    
    gc.collect()

np.save(f'/homedata/dveytia/Product_1_data/outputs/predictions_data/{targetVar}_{n_folds}fold_data_pred_ids_v{v}.npz',unseen_df["id"])