import pandas as pd
import numpy as np

"""
here I want to compile all the oro predictions into one csv

NEED TO ADD:
Make sure that the predictions are set to NA where prediction boundaries were in place. Because model not suited for that data. 
"""

targetVar = "oro_type"
v=2 # version of the script
dataFolder = '/homedata/dveytia/Product_1_data'

## Import predictions

## Multilabel variables
# use MRE-Ocean and MRE-Located
MRE_pred = pd.read_csv(f'{dataFolder}/outputs/predictions-compiled/oro_type.MRE_v2_MreLO_predictions.csv')
keepCols = ['id']
keepCols.extend([col for col in MRE_pred.columns if any(label in col for label in ['Located','Ocean'])])
MRE_pred = MRE_pred[keepCols]
print(MRE_pred.columns)

# use CCS, CDR-BC, MRE-Bio, Efficiency
NotMRE_pred = pd.read_csv(f'{dataFolder}/outputs/predictions-compiled/oro_type_v2_notMRE_predictions.csv')
keepCols = ['id']
keepCols.extend([col for col in NotMRE_pred.columns if any(label in col for label in ['CCS', 'CDR-BC', 'MRE-Bio', 'Efficiency'])])
NotMRE_pred = NotMRE_pred[keepCols]
print(NotMRE_pred.columns)

## Binary variables (only one variable each so no need to subset columns)
# Use CDR-OAE
OAE_pred = pd.read_csv(f'{dataFolder}/outputs/predictions-compiled/oro_type.CDR-OAE_1_predictions.csv')

# Use CDR-BioPump
BioPump_pred = pd.read_csv(f'{dataFolder}/outputs/predictions-compiled/oro_type.CDR-BioPump_1_predictions.csv')



## Join all predictions together
dfs = [MRE_pred, NotMRE_pred, OAE_pred, BioPump_pred]
oro_pred = dfs[0]
for df_ in dfs[1:]:
    oro_pred = oro_pred.merge(df_, on='id', how="outer")
print(oro_pred.columns)
print(len(oro_pred))

## How many ids are predicted relevant for other ORO types when MRE-Located or MRE-Ocean?
tempDf = oro_pred[(oro_pred['oro_type.MRE-Located - mean_prediction'] >= 0.5) | (oro_pred['oro_type.MRE-Ocean - mean_prediction'] >= 0.5)]

def countRelevant(col):
    col = col.to_numpy()
    nRel = np.where(col>=0.5,1,0)
    return nRel

nonMreCols = [
    'oro_type.MRE-Bio - mean_prediction',
    'oro_type.Efficiency - mean_prediction',
    'oro_type.CCS - mean_prediction',
    'oro_type.CDR-BC - mean_prediction',
    'oro_type.CDR-OAE - mean_prediction',
    'oro_type.CDR-BioPump - mean_prediction'
]
tempDf[nonMreCols].apply(countRelevant, axis=1).sum()

mreIdx = tempDf.index

## save the subsetted predictions for future labels predictions
oro_pred.iloc[mreIdx].to_csv(rf'{dataFolder}/outputs/predictions-compiled/{targetVar}_{v}_MRE_predictions.csv',index=False)

oro_pred.iloc[list(set(oro_pred.index) - set(mreIdx))].to_csv(rf'{dataFolder}/outputs/predictions-compiled/{targetVar}_{v}_notMRE_predictions.csv',index=False)


"""
array([ 615, 4561, 2202, 2860,  238,  470])

Not many predicted relevant alongside MRE, so I think it's ok to keep. I was worried maybe if some labels were predicted high that the classifier was not good at extrapolating but it seems ok. I was going to set the other labels to NA when MRE-Located or MRE-Ocean >=0.5 but I don' think this is necessary
"""




oro_pred.to_csv(rf'{dataFolder}/outputs/predictions-compiled/{targetVar}_{v}_predictions.csv',index=False)

