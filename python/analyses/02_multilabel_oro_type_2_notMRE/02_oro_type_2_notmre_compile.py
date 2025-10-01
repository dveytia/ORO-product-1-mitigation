import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################# Change INPUTS ##################


targetVar = "oro_type" # name of variable
v='2_notMRE' # Version of the script
dataFolder = '/homedata/dveytia/Product_1_data'
codedVariablesTxt = f'{dataFolder}/data/articleAnswers_notMRE_formatted_2025-05-26.txt' # file with coded variables
n_folds = 5


################# Load documents and format ##################

df = pd.read_csv(codedVariablesTxt, delimiter='\t') 



################# using unseen_ids file to compile preds #####################
unseen_ids = pd.DataFrame(np.load(f'{dataFolder}/outputs/predictions_data/{targetVar}_{n_folds}fold_data_pred_ids_v{v}.npz.npy'))
# unseen_ids = df.loc[unseen_index,"id"]
unseen_ids.columns=["id"]

targets = [x for x in df.columns if targetVar in x] 
targets = [x for x in targets if x not in [f'{targetVar}.MRE-Located',f'{targetVar}.MRE-Ocean']] # But remove MRE Ocean and Located
# Should I add an 'other' option? Yes because on first run, predicted relevant to everything
df[f'{targetVar}-Absent'] = df[targets].eq(0).all(axis=1).astype(int)
targets.append(f'{targetVar}-Absent')

y_preds = [ np.zeros((len(unseen_ids),n_folds)) for x in range(len(targets))]



for k in range(n_folds):
    y_pred = np.load(f'{dataFolder}/outputs/predictions/{targetVar}_y_preds_{n_folds}fold_data_v{v}_k{k}.npz.npy')
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
    

unseen_ids.to_csv(f'{dataFolder}/outputs/predictions-compiled/{targetVar}_v{v}_predictions.csv',index=False) #Saves .csv file, change file path



## EXPLORE PREDICTIONS -----------------------------


# Step 1: Identify columns with 'mean_prediction'
mean_cols = [col for col in unseen_ids.columns if 'mean_prediction' in col]

# Step 2: Create a boolean DataFrame: True if prediction ≥ 0.5
relevance_flags = unseen_ids[mean_cols] >= 0.5

# Step 3: Count how many labels are predicted relevant per row
relevance_flags.sum() # relevant labels per colum
relevant_counts = relevance_flags.sum(axis=1)

# Step 4: Count how many rows fall into each category
only_one_relevant = (relevant_counts == 1).sum()
two_relevant = (relevant_counts == 2).sum()
three_relevant = (relevant_counts == 3).sum()
none_relevant = (relevant_counts == 0).sum()

# Output the results
print(f"Documents with only one relevant label: {only_one_relevant}")
print(f"Documents with two labels relevant: {two_relevant}")
print(f"Documents with three labels relevant: {three_relevant}")
print(f"Documents with no relevant labels: {none_relevant}")

# Column totals
print("Sums of relevance predictions for each label:")
print(relevance_flags.sum(axis=0))

"""
Documents with only one relevant label: 60717
Documents with two labels relevant: 7
Documents with three labels relevant: 0
Documents with no relevant labels: 1596
Sums of relevance predictions for each label:
oro_type.MRE-Located - mean_prediction    17021
oro_type.MRE-Ocean - mean_prediction      13776
oro_type.MRE-Absent - mean_prediction     29934

"""

## Plot precision recall curve? -----------------
from sklearn.metrics import classification_report, f1_score, recall_score, precision_recall_curve, PrecisionRecallDisplay

shared_ids_df = seen_df.merge(unseen_ids, on='id')

for t in targets:
    y_test = shared_ids_df[t].to_list()
    preds = shared_ids_df[f'{t} - mean_prediction'].to_list()
    prCurve = PrecisionRecallDisplay.from_predictions(y_test, preds)
    precisions, recalls, thresholds = precision_recall_curve(y_test, preds)
    best_threshold = thresholds[np.argmax(recalls >= 0.7)]
    print(f'Best threshold with recall >= 0.7: {best_threshold}')
    print(f"{len([x for x in preds if x > best_threshold])} predicted relevant / {len(preds)}")

    # Plot showing the best threshold as a dot
    precisions[np.argmin(recalls >= 0.7)]
    recalls[np.argmin(recalls >= 0.7)]
    bt = str(round(best_threshold, 3))
    nr = str(round(len([x for x in preds if x >= best_threshold])/len(preds)*100))
    plt.figure(figsize=(6, 4))
    prCurve.plot()
    plt.scatter(recalls[np.argmin(recalls >= 0.7)], precisions[np.argmin(recalls >= 0.7)], color='red', label=f"Selected Threshold {bt}")
    plt.text(min(recalls), min(precisions), f'{nr}% predicted relevant', ha='left', wrap=True)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("FNN Precision-Recall Curve")
    plt.legend()
    plt.show()
    # plt.savefig(f"{dataFolder}/figures/supplemental/{targetVar}_v{v}_PrecRecallCurve.png", dpi=300, bbox_inches='tight')




# plt.figure(figsize=(6, 4))
# prCurve.plot()
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.legend()
# # plt.savefig("/home/dveytia/IPython_Notebooks/Product_2/tmpfigs/umapRandForestPrecRecallCurve.png", dpi=300, bbox_inches='tight')
# plt.savefig("/homedata/dveytia/Product_2_data/figures/supplemental/umapRandForestPrecRecallCurve.png", dpi=300, bbox_inches='tight')
# plt.show()

