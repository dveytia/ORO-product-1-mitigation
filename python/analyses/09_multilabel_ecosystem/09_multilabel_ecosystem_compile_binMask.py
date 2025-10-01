import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################# Change INPUTS ##################

targetVar = "ecosystem" # name of variable
v='1' # Version of the script
dataFolder = '/homedata/dveytia/Product_1_data'
n_folds = 5

# Mask the ecosystem type predictions by binary ecosystem
mask_ecosys_binary = pd.read_csv(rf'{dataFolder}/outputs/predictions-compiled/ecosystem_binary_1_predictions.csv')
mask_ecosys_binary = mask_ecosys_binary[['id','ecosystem_binary - mean_prediction']]
# mask ids -- set these to na
mask_ids = mask_ecosys_binary.loc[mask_ecosys_binary['ecosystem_binary - mean_prediction']>=0.5,'id'].to_list()

################# using unseen_ids file to compile preds #####################
unseen_ids= pd.DataFrame(np.load(f'{dataFolder}/outputs/predictions_data/{targetVar}_{n_folds}fold_data_pred_ids_v{v}.npz.npy')) 
unseen_ids.columns=["id"]

targets = ['ecosystem.Microalgae',
 'ecosystem.Macroalgae',
 'ecosystem.Seagrass',
 'ecosystem.Salt marsh',
 'ecosystem.Mangrove']

y_preds = [ np.zeros((len(unseen_ids),n_folds)) for x in range(len(targets))]

all_cols = ['id']

for k in range(n_folds):
    y_pred = np.load(f'{dataFolder}/outputs/predictions/{targetVar}_y_preds_{n_folds}fold_data_v{v}_k{k}.npz.npy')

    # set all columns of an mask id index to na?
    mask = unseen_ids['id'].isin(mask_ids).to_numpy()  # True if should keep, False if mask
    y_pred[~mask, :] = np.nan
    # Or just remove these indices entirely?
    
    for i in range(len(targets)):
        y_preds[i][:,k] = y_pred[:,i]
        
for i in range(len(targets)):
    mean_pred = np.nanmean(y_preds[i], axis=1)
    std_pred = np.nanstd(y_preds[i], axis=1)

    preds_upper = np.minimum(mean_pred + std_pred, 1)
    preds_lower = np.maximum(mean_pred - std_pred, 0)
    
    t = targets[i]
    
    unseen_ids[f'{t} - mean_prediction'] = mean_pred
    unseen_ids[f'{t} - std_prediction'] = std_pred
    unseen_ids[f'{t} - lower_pred'] = preds_lower
    unseen_ids[f'{t} - upper_pred'] = preds_upper
    
    print(targets[i]) 
    print(unseen_ids.sort_values(f'{t} - mean_prediction',ascending=False).head())
    






unseen_ids.to_csv(f'{dataFolder}/outputs/predictions-compiled/{targetVar}_binaryMask_{v}_predictions.csv',index=False) #Saves .csv file, change file path


## summarize
unseen_ids[[x for x in unseen_ids.columns if '- mean' in x]].apply(lambda x: np.where(x>0.5,1,0).sum())
"""
ecosystem.Microalgae - mean_prediction    49767
ecosystem.Macroalgae - mean_prediction     2880
ecosystem.Seagrass - mean_prediction        578
ecosystem.Salt marsh - mean_prediction      824
ecosystem.Mangrove - mean_prediction       1552

I think there is a problem here, because overall len(unseen_ids) == 58952 ---- can't all be about microalgae?!
Maybe I need to add another binary classifer of ecosystem vs no ecosystem to overlay?

After the mask, much better:
ecosystem.Microalgae - mean_prediction     600
ecosystem.Macroalgae - mean_prediction     295
ecosystem.Seagrass - mean_prediction       534
ecosystem.Salt marsh - mean_prediction     606
ecosystem.Mangrove - mean_prediction      1424
"""



