import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

######## Change inputs ###########
targetVar = "oro_screen_update"
dataFolder = '/homedata/dveytia/Product_1_data'
screenDecisionsTxt = f'{dataFolder}/data/all-screen-results_screenExcl-codeIncl.txt'
unseenTxt = f'{dataFolder}/data/unique_references_UPDATE_13-05-2025.txt'
k = 10 # For screening predictions there are 10 folds but for others only 5

######## Load files, change paths #################
seen_df = pd.read_csv(screenDecisionsTxt, delimiter='\t')
seen_df['seen']=1
seen_df = seen_df.rename(columns={'include_screen':'relevant','analysis_id':'id'})
seen_df['relevant']=seen_df['relevant'].astype(int)

def map_values(x): 
    value_map = {
        "random": 1,
        "relevance sort": 0,
        "test list": 0,
        "supplemental coding": 0
    }
    return value_map.get(x, "NaN")

seen_df['random_sample'] = seen_df['sample_screen'].apply(map_values)


unseen_df = pd.read_csv(unseenTxt, delimiter='\t') 
unseen_df.rename(columns={'analysis_id':'id'}, inplace=True)
unseen_df['seen']=0
unseen_df=unseen_df.dropna(subset=['abstract']).reset_index(drop=True)
unseen_df = unseen_df[['id', 'title', 'abstract', 'keywords', 'seen']]


df = (pd.concat([seen_df,unseen_df])
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)


seen_index = df[df['seen']==1].index
unseen_index = df[df['seen']==0].index


############## Load files for predictions, change file paths ##################
k=10
if k==10:
    y_preds = np.zeros((len(unseen_index),10))

    for k in range(10):
        y_pred = np.load(f'{dataFolder}/outputs/predictions/{targetVar}_y_preds_10fold_{k}.npz.npy')[:,0]#[:len(unseen_index),0]
        y_preds[:,k] = y_pred
        print(np.where(y_pred>0.5,1,0).sum())    
else:
    y_preds = np.zeros((len(unseen_index),5))

    for k in range(5):
        y_pred = np.load(f'{dataFolder}/outputs/predictions/{targetVar}_predictions_5fold_{k}.npz.npy')[:,0]#[:len(unseen_index),0]
        y_preds[:,k] = y_pred
        print(np.where(y_pred>0.5,1,0).sum())
    
mean_pred = np.mean(y_preds, axis=1)
std_pred = np.std(y_preds, axis=1)

preds_upper = np.minimum(mean_pred + std_pred, 1)
preds_lower = np.maximum(mean_pred - std_pred, 0)

df.loc[unseen_index,'0 - relevance - mean_prediction'] = mean_pred
df.loc[unseen_index,'0 - relevance - std_prediction'] = std_pred
df.loc[unseen_index,'0 - relevance - lower_pred'] = preds_lower
df.loc[unseen_index,'0 - relevance - upper_pred'] = preds_upper


cols = [
    "0 - relevance - mean_prediction",
    "0 - relevance - std_prediction",
    "0 - relevance - lower_pred",
    "0 - relevance - upper_pred"  
]
for c in cols:
    df.loc[seen_index,c] = df.loc[seen_index,"relevant"]

#df[["id"]+cols].to_csv(r'03_Binary-AllText-NewApproach\1_document_relevance.csv',index=False) # Save predictions to csv file

df[["id"]+cols].to_csv(f'{dataFolder}/outputs/predictions-compiled/{targetVar}_predictions.csv',index=False) #Saves .csv file, change file path


for i, row in df[df['0 - relevance - mean_prediction']>0.5].sample(10).iterrows(): # Check a few
    print(row.title)
    print(row[cols])
    
    
df[df['0 - relevance - mean_prediction']>0.5].shape # Shows number of inclusions looking at mean predictions

#################### Create figure for inclusions ####################
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
plt.savefig(f'{dataFolder}/figures/{targetVar}_predictions_unseen.png',bbox_inches="tight") # Save plot, change file paths



##############################################################################



# ################# using unseen_ids file to compile preds #####################
# unseen_ids= pd.DataFrame(np.load(r'03_Binary-AllText-NewApproach\predictions_excl\unseen_ids.npz.npy')) #Change file path
# unseen_ids.columns=["id"]

# k = 10

# if k==10:
#     y_preds = np.zeros((len(unseen_ids),10))

#     for k in range(10):
#         y_pred = np.load(rf'03_Binary-AllText-NewApproach\predictions_excl\y_preds_10fold_{k}.npz.npy')[:,0]#[:len(unseen_index),0]
#         y_preds[:,k] = y_pred
#         print(np.where(y_pred>0.5,1,0).sum())    
# else:
#     y_preds = np.zeros((len(unseen_index),5))

#     for k in range(5):
#         y_pred = np.load(rf'03_Binary-AllText-NewApproach\predictions_excl\y_preds_10fold_{k}.npz.npy')[:,0]#[:len(unseen_index),0]
#         y_preds[:,k] = y_pred
#         print(np.where(y_pred>0.5,1,0).sum())
    
# mean_pred = np.mean(y_preds, axis=1)
# std_pred = np.std(y_preds, axis=1)

# preds_upper = np.minimum(mean_pred + std_pred, 1)
# preds_lower = np.maximum(mean_pred - std_pred, 0)

# unseen_ids['0 - relevance - mean_prediction'] = mean_pred
# unseen_ids['0 - relevance - std_prediction'] = std_pred
# unseen_ids['0 - relevance - lower_pred'] = preds_lower
# unseen_ids['0 - relevance - upper_pred'] = preds_upper

# unseen_ids_sorted = unseen_ids.sort_values('id').reset_index(drop=True)
# for i, row in unseen_ids_sorted[unseen_ids_sorted['0 - relevance - mean_prediction']>0.5].sample(10).iterrows():
#     print(row.title)
#     print(row[cols])
           

# unseen_ids.to_csv(r'03_Binary-AllText-NewApproach\1_document_relevance_v2.csv',index=False) # Save file, change path

