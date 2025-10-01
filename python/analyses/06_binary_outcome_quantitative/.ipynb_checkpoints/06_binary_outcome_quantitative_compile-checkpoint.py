import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################# Change INPUTS ##################
targetVar = 'outcome_quantitative' # name of variable
v='1' # Version of the script
dataFolder = '/homedata/dveytia/Product_1_data'
n_folds = 5 


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


########### check results -- how many for each oro type? ###########3
oro_pred = pd.read_csv(f'{dataFolder}/outputs/predictions-compiled/oro_type_2_predictions.csv')
tempDf = unseen_ids.merge(oro_pred, on = 'id', how = 'left')
tempDf = tempDf[[x for x in tempDf.columns if 'mean_prediction' in x]]
tempDf.loc[tempDf['outcome_quantitative - mean_prediction']>=0.5,[x for x in tempDf.columns if 'oro_type' in x]].apply(lambda x: np.where(x>0.5,1,0)).sum()

"""
Results:

oro_type.MRE-Located - mean_prediction    12138
oro_type.MRE-Ocean - mean_prediction      17271
oro_type.CCS - mean_prediction             1612
oro_type.CDR-BC - mean_prediction          2528
oro_type.MRE-Bio - mean_prediction          312
oro_type.Efficiency - mean_prediction      3199
oro_type.CDR-OAE - mean_prediction          205
oro_type.CDR-BioPump - mean_prediction      320

"""







