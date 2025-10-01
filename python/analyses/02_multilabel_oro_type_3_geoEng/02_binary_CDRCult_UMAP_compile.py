import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Note this version compiles from the UMAP results, but only for folds 0 and 2, because fold 1 did not perform well:

   ROCAUC	F1	precision	recall	accuracy	fold	Model
0	0.4	0.571429	0.444444	0.8	0.4	0	oro_type.CDR-Cult
1	0.5	0.000000	0.000000	0.0	0.5	1	oro_type.CDR-Cult
2	0.6	0.666667	0.571429	0.8	0.6	2	oro_type.CDR-Cult

"""

################# Change INPUTS ##################
targetVar = "oro_type.CDR-Cult" # name of variable
v='3_UMAP' # Version of the script
dataFolder = '/homedata/dveytia/Product_1_data'
n_folds = 3

unseen_ids = pd.DataFrame(np.load(f"/homedata/dveytia/Product_1_data/outputs/predictions_data/{targetVar}_{n_folds}fold_data_pred_ids_v{v}.npz.npy"))

unseen_ids.columns=["id"]


y_preds = np.zeros((len(unseen_ids),n_folds))

for k in range(n_folds):
    y_pred = np.load(rf"{dataFolder}/outputs/predictions/{targetVar}_y_preds_{n_folds}fold_data_v{v}_k{k}.npz.npy")
    y_preds[:,k] = y_pred
    print(np.where(y_pred>0.5,1,0).sum())

y_preds = np.delete(y_preds, 1, axis=1)
y_preds.shape

mean_pred = np.mean(y_preds, axis=1)
std_pred = np.std(y_preds, axis=1)

preds_upper = np.minimum(mean_pred + std_pred, 1)
preds_lower = np.maximum(mean_pred - std_pred, 0)

unseen_ids[f'{targetVar} - mean_prediction'] = mean_pred
unseen_ids[f'{targetVar} - std_prediction'] = std_pred
unseen_ids[f'{targetVar} - lower_pred'] = preds_lower
unseen_ids[f'{targetVar} - upper_pred'] = preds_upper

unseen_ids.to_csv(rf'{dataFolder}/outputs/predictions-compiled/{targetVar}_{v}_predictions.csv',index=False)


### Summarize
print(np.where(mean_pred>=0.5,1,0).sum()) # 57189




##################### plot ####################

## simple density plot
sns.set_style('whitegrid')
sns.kdeplot(np.array(mean_pred), bw=0.5)

# upper mode at 0.75 -- check
print(np.where(mean_pred>=0.75,1,0).sum()) # 34293
# do I still get full recall at this?


## Callaghan plot
fig, ax = plt.subplots(dpi=150)

b = np.mean(y_preds, axis = 1)
idx = b.argsort()
y_preds_sorted = np.take(y_preds, idx, axis=0)

mean_pred = np.mean(y_preds_sorted, axis=1)
std_pred = np.std(y_preds_sorted, axis=1)

ax.plot(mean_pred, color='r', label="Mean")

preds_upper = np.minimum(mean_pred + std_pred, 1)
preds_lower = np.maximum(mean_pred - std_pred, 0)

ax.fill_between(range(len(mean_pred)), preds_upper, preds_lower, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

lb = preds_upper[np.where(preds_upper>0.5)].shape[0]
ub = preds_lower[np.where(preds_lower>0.5)].shape[0]
mb = mean_pred[np.where(mean_pred>0.5)].shape[0]

s = f'{mb:,} ({ub:,}-{lb:,})\n relevant documents predicted'

ax.plot([np.argwhere(preds_upper>0.5)[0][0]*0.75,np.argwhere(preds_upper>0.5)[0][0]],[0.6,0.5],c="grey",ls="--")
ax.plot([np.argwhere(preds_upper>0.5)[0][0]*0.75,np.argwhere((preds_lower>0.5) & (preds_lower < 0.501))[-1][0]],[0.6,0.5],c="grey",ls="--")
ax.text(np.argwhere(preds_upper>0.5)[0][0]*0.75,0.6,s,ha="right",va="bottom",bbox=props)

ax.set_xlabel('Documents')
ax.set_ylabel('Predicted relevance')

ax.legend()
plot.show()


### Print out a file with relevant articles to check?
# After checking, these results are no good -- all the inclusions have the same prediction of 0.667 and the articles don't seem relevant to macroalage cultivation

# get unseen data
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
unseen_df = unseen_df[(unseen_df['oro_type.MRE-Located - mean_prediction']<0.5)]
unseen_df = unseen_df[(unseen_df['oro_type.MRE-Ocean - mean_prediction']<0.5)]
unseen_df['id'] = unseen_df['id'].astype(int)


# merge in with relevant articles
checkArticles = unseen_ids.loc[unseen_ids[f'{targetVar} - mean_prediction']>=0.75]
checkArticles['id'] = checkArticles['id'].astype(int)
checkArticles = checkArticles.merge(unseen_df, how='inner', on='id')
print(checkArticles.shape) # Now only 10010 articles

# save
checkArticles.to_excel(rf'{dataFolder}/outputs/predictions-compiled/{targetVar}_{v}_predictionsCHECK.xlsx', index=False)

