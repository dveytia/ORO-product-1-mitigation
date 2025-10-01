import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################# Change INPUTS ##################
targetVar = 'ecosystem_binary' # name of variable
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



# unseen_ids.to_csv(rf'{dataFolder}/outputs/predictions-compiled/{targetVar}_{v}_predictions.csv',index=False)


########### check results -- how many primary vs not for each oro type? ###########3
oro_pred = pd.read_csv(f'{dataFolder}/outputs/predictions-compiled/oro_type_2_predictions.csv')
tempDf = unseen_ids.merge(oro_pred, on = 'id', how = 'left')
tempDf = tempDf[[x for x in tempDf.columns if 'mean_prediction' in x]]
tempDf.loc[tempDf[f'{targetVar} - mean_prediction']>=0.5,[x for x in tempDf.columns if 'oro_type' in x]].apply(lambda x: np.where(x>=0.5,1,0)).sum()

"""
Result:

oro_type.MRE-Located - mean_prediction      15
oro_type.MRE-Ocean - mean_prediction      5907
oro_type.CCS - mean_prediction              15
oro_type.CDR-BC - mean_prediction         2781
oro_type.MRE-Bio - mean_prediction         439
oro_type.Efficiency - mean_prediction        6
oro_type.CDR-OAE - mean_prediction          40
oro_type.CDR-BioPump - mean_prediction     321

Buuuttt in the coded sample this is the representation of ecocystem type by oro type. Maybe only include the ORO types: oro_type.MRE-Bio, oro_type.CDR-BioPump, oro_type.CDR-Cult, oro_type.CDR-BC

oro_type.CDR-BioPump     57
oro_type.MRE-Ocean        9
oro_type.MRE-Bio         95
oro_type.CDR-Cult        47
oro_type.CDR-OAE         16
oro_type.CDR-BC         135
oro_type.Efficiency       1
oro_type.MRE-Located      2
oro_type.CDR-Other        4
oro_type.CCS              5

"""
included_oro_types = [
    'oro_type.MRE-Bio - mean_prediction',
    'oro_type.CDR-BioPump - mean_prediction',
    'oro_type.CDR-BC - mean_prediction',
]

# Boolean mask: True if any of the selected ORO types >= 0.5
retain_mask = tempDf[included_oro_types].ge(0.5).any(axis=1)

# Set all prediction columns in unseen_ids to NaN if mask is False
prediction_cols = [f'{targetVar} - mean_prediction',
                   f'{targetVar} - std_prediction',
                   f'{targetVar} - lower_pred',
                   f'{targetVar} - upper_pred']

unseen_ids.loc[~retain_mask, prediction_cols] = np.nan

################### Save final predictions ###################
unseen_ids.to_csv(f'{dataFolder}/outputs/predictions-compiled/{targetVar}_{v}_predictions.csv', index=False)






