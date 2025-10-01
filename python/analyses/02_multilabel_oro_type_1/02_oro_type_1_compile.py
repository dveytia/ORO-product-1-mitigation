import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################# Change INPUTS ##################


# relevanceTxt = '/home/dveytia/ORO-map-relevance/outputs/predictions-compiled/1_document_relevance_13062023.csv'

targetVar = "oro_type" # name of variable
v='1' # Version of the script
dataFolder = '/homedata/dveytia/Product_1_data'
codedVariablesTxt = f'{dataFolder}/data/articleAnswers_formatted_2025-04-26.txt' # file with coded variables
unseenTxt = f'{dataFolder}/data/all_unseen_mitigation_oros.txt' # change to unique_references2.txt?
n_folds = 3 # 5


################# Load documents and format ##################

# Load seen documents
seen_df = pd.read_csv(codedVariablesTxt, delimiter='\t')
seen_df = seen_df.rename(columns={'analysis_id':'id'})
seen_df['seen']=1

# Load unseen documents 
unseen_df = pd.read_csv(unseenTxt, delimiter='\t') 
#unseen_df = unseen_df.rename(columns={'analysis_id':'id'})
unseen_df=unseen_df.dropna(subset=['abstract']).reset_index(drop=True)

# Choose which prediction boundaries to apply. 
# In final version will use the screening model:
"""
pred_df = pd.read_csv(relevanceTxt)  # Load prediction relevance
unseen_df = unseen_df.merge(pred_df, how="left") # merge
unseen_df['seen']=0
unseen_df = unseen_df[unseen_df['0 - relevance - upper_pred']>=0.5] # Choose which predictiction boundaries to apply
"""
# but for now use predictions from ORO map mitigation variables
unseen_df = unseen_df[(unseen_df['oro_any.M_Renewables - mean_prediction']>=0.5) | (unseen_df['oro_any.M_Increase_efficiency - mean_prediction']>=0.5) | (unseen_df['oro_any.M_CO2_removal_or_storage - mean_prediction']>=0.5)]
unseen_df['seen']=0


# Remove seen articles from unseen 
unseen_df = unseen_df[~(unseen_df.id.isin(seen_df.id))]

# Concatenate seen and unseen
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


################# using unseen_ids file to compile preds #####################
unseen_ids= pd.DataFrame(np.load(f'/homedata/dveytia/SPATMAN_data/outputs/predictions_data/{targetVar}_data_pred_ids.npy')) #Change file path
unseen_ids = pd.DataFrame(np.load(f'/homedata/dveytia/Product_1_data/outputs/predictions_data/{targetVar}_{n_folds}fold_data_pred_ids_v{v}_k2.npy'))
# unseen_ids = df.loc[unseen_index,"id"]
unseen_ids.columns=["id"]

targets = [x for x in df.columns if targetVar in x]

y_preds = [ np.zeros((len(unseen_ids),n_folds)) for x in range(len(targets))]

all_cols = ['id']

for k in range(n_folds):
    y_pred = np.load(f'/homedata/dveytia/Product_1_data/outputs/predictions/{targetVar}_y_preds_{n_folds}fold_data_v{v}_k{k}.npz.npy')
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
    

unseen_ids.to_csv(f'/homedata/dveytia/SPATMAN_data/outputs/predictions-compiled/{targetVar}_v{v}_predictions.csv',index=False) #Saves .csv file, change file path


