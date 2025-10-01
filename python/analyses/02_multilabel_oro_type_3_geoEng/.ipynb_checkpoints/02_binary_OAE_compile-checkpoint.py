import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################# Change INPUTS ##################
targetVar = "oro_type.CDR-OAE" # name of variable
v='1' # Version of the script
dataFolder = '/homedata/dveytia/Product_1_data'
codedVariablesTxt = f'{dataFolder}/data/articleAnswers_formatted_2025-04-26.txt' # file with coded variables
unseenTxt = f'{dataFolder}/data/all_unseen_mitigation_oros.txt' # change to unique_references2.txt?
n_folds = 5 


############################# Load data ###############################
######################## Change file paths x3 #########################

# Load seen documents
seen_df = pd.read_csv(codedVariablesTxt, delimiter='\t') 
#seen_df = seen_df.rename(columns={'analysis_id':'id'})
seen_df['seen']=1

# Keep screening decisions in
# seen_df = seen_df.loc[seen_df["relevant"] == 1,]
seen_df = seen_df.dropna(subset=["abstract"]) # if any abstracts are NA, remove




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

## Choose prediction bounndaries -- NOT MRE
pred_MRE = pd.read_csv(f'{dataFolder}/outputs/predictions-compiled/oro_type.MRE_v2_MreLO_predictions.csv')
unseen_df = unseen_df.merge(pred_MRE, how = "left")
unseen_df = unseen_df[(unseen_df['oro_type.MRE-Located - mean_prediction']<0.5) | (unseen_df['oro_type.MRE-Ocean - mean_prediction']<0.5)]



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

################################## ######################################
# # Unseen ids from np file
unseen_ids = pd.DataFrame(np.load(f"{dataFolder}/outputs/predictions_data/{targetVar}_{n_folds}fold_data_pred_ids_v{v}.npz.npy"))
# unseen_ids = pd.DataFrame(df.loc[unseen_index,'id'])
unseen_ids.columns=["id"]


y_preds = np.zeros((len(unseen_ids),n_folds))

for k in range(n_folds):
    y_pred = np.load(rf"{dataFolder}/outputs/predictions/{targetVar}_y_preds_{n_folds}fold_data_v{v}_k{k}.npz.npy")[:,0]#Change file path
    y_preds[:,k] = y_pred
    print(np.where(y_pred>0.5,1,0).sum())


        
mean_pred = np.mean(y_preds, axis=1)
std_pred = np.std(y_preds, axis=1)

preds_upper = np.minimum(mean_pred + std_pred, 1)
preds_lower = np.maximum(mean_pred - std_pred, 0)

unseen_ids[f'{targetVar} - mean_prediction'] = mean_pred
unseen_ids[f'{targetVar} - std_prediction'] = std_pred
unseen_ids[f'{targetVar} - lower_pred'] = preds_lower
unseen_ids[f'{targetVar} - upper_pred'] = preds_upper



unseen_ids.to_csv(rf'{dataFolder}/outputs/predictions-compiled/{targetVar}_{v}_predictions.csv',index=False)










