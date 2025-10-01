'''
Multilabel predictions

Multilabel oro_branch on predicted relevant documents
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataFolder = '/homedata/dveytia/Product_1_data'
targetVar = "oro_branch"
v = 'update'
n_folds = 5



# Load seen documents
seen_df = pd.read_csv(f'{dataFolder}/data/all-coding-format-distilBERT-simplifiedMore.txt', delimiter='\t')
seen_df = seen_df.rename(columns={'analysis_id':'id'})
seen_df['seen']=1


# Load unseen documents and merge
unseen_df = pd.read_csv(f'{dataFolder}/data/unique_references_UPDATE_13-05-2025.txt', delimiter='\t')
unseen_df = unseen_df.rename(columns={'analysis_id':'id'})
unseen_df=unseen_df.dropna(subset=['abstract']).reset_index(drop=True)

pred_df = pd.read_csv(f'{dataFolder}/outputs/predictions-compiled/oro_screen_update_predictions.csv')

unseen_df = unseen_df.merge(pred_df, how="left")
unseen_df['seen']=0

# Choose which predictiction boundaries to apply
unseen_df = unseen_df[unseen_df['0 - relevance - mean_prediction']>=0.5]


# Concatenate seen and unseen
df = (pd.concat([seen_df,unseen_df])
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)

df["keywords"] = 'Keywords: '+ df["keywords"]
df["text"] = df[["title", "abstract", "keywords"]].agg(
    lambda x: ' '.join(x.dropna()), axis=1
)


seen_index = df[df['seen']==1].index
unseen_index = df[df['seen']==0].index

print("Dataset has been re-formatted and is ready")


################################## ######################################
# Unseen ids from np file
# unseen_ids = pd.DataFrame(np.load(r'05_Multilabel_1_oro_branch\predictions_oro_branch_mean\oro_branch_pred_ids.npy'))
unseen_ids = pd.DataFrame(np.load(f"{dataFolder}/outputs/predictions_data/{targetVar}_{v}_unseen_ids.npz.npy"))
unseen_ids.columns=["id"]
unseen_ids['id'] = unseen_ids['id']+264 # Fix for indexing error


targets = [x for x in df.columns if targetVar in x]

        #Use if using unseen_ids to compile
y_preds = [ np.zeros((len(unseen_ids),5)) for x in range(len(targets))]

all_cols = ['id']

for k in range(n_folds):
    # y_pred = np.load(rf"05_Multilabel_1_oro_branch\predictions_oro_branch_mean\y_preds_5fold_oro_branch_{k}.npz.npy")
    y_pred = np.load(rf"{dataFolder}/outputs/predictions/{targetVar}_{v}_y_preds_{n_folds}fold_{k}.npz.npy")
    
    for i in range(len(targets)):
        y_preds[i][:,k] = y_pred[:,i]
        
for i in range(len(targets)):
    mean_pred = np.mean(y_preds[i], axis=1)
    std_pred = np.std(y_preds[i], axis=1)

    preds_upper = np.minimum(mean_pred + std_pred, 1)
    preds_lower = np.maximum(mean_pred - std_pred, 0)
    
    t = targets[i]
    
    unseen_ids[f'{t} - mean_prediction'] = mean_pred
    unseen_ids[f'{t} - std_prediction'] = std_pred
    unseen_ids[f'{t} - lower_pred'] = preds_lower
    unseen_ids[f'{t} - upper_pred'] = preds_upper
    
    print(targets[i]) 
    print(unseen_ids.sort_values(f'{t} - mean_prediction',ascending=False).head())
    

unseen_ids.to_csv(rf'{dataFolder}/outputs/predictions-compiled/{targetVar}_{v}_predictions.csv',index=False)


    
#  ######## # Use if using unseen_index (i.e. df = seen + unseen)
# y_preds = [np.zeros((len(unseen_index),5)) for x in range(len(targets))]

# all_cols = ['id']

# for k in range(5):
#     y_pred = np.load(rf"05_Multilabel_1_oro_branch\predictions_oro_branch_mean\y_preds_5fold_oro_branch_{k}.npz.npy")
    
#     for i in range(len(targets)):
#         y_preds[i][:,k] = y_pred[:,i]
#         print(np.where(y_pred[:,i]>0.5,1,0).sum())
# for i in range(len(targets)):
#     mean_pred = np.mean(y_preds[i], axis=1)
#     std_pred = np.std(y_preds[i], axis=1)
    
#     preds_upper = np.minimum(mean_pred + std_pred, 1)
#     preds_lower = np.maximum(mean_pred - std_pred, 0)
    
#     print(targets[i])
#     t = targets[i]
#     print(np.where(mean_pred>0.5,1,0).sum())
#     df.loc[unseen_index,f'{t} - mean_prediction'] = mean_pred
#     df.loc[unseen_index,f'{t} - std_prediction'] = std_pred
#     df.loc[unseen_index,f'{t} - lower_pred'] = preds_lower
#     df.loc[unseen_index,f'{t} - upper_pred'] = preds_upper
    
#     cols = [
#         f'{t} - mean_prediction',
#         f'{t} - std_prediction',
#         f'{t} - lower_pred',
#         f'{t} - upper_pred'  
#     ]
#     all_cols += cols
#     for c in cols:
#         df.loc[seen_index,c] = df.loc[seen_index,t]
        
#     print(df.sort_values(f'{t} - mean_prediction',ascending=False).head())
    
    
# df.to_csv(r'05_Multilabel_1_oro_branch\full-oro_branch_predictions_mean.csv',index=False)


# ################# a few checks ################
# duplicates = df[df.duplicated(subset=['id'], keep=False)]
# df_sorted=duplicates.sort_values('id')

# duplicate_seen=df_sorted[df_sorted['seen']==1]
# duplicate_unseen=df_sorted[df_sorted['seen']==0]
# for i, row in df_sorted[df_sorted['oro_branch.Mitigation']==1].sample(20).iterrows():
#    # print(row.title)
#     #print(row.abstract)
#     print(row.id)
#     print(row["oro_branch.Mitigation - mean_prediction"])
