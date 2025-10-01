import sys
print(sys.version)
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()
rank_i = rank


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import ast
import time
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import tensorflow_addons as tfa
import os
import gc

t0 = time.time()

################# Change INPUTS ##################
targetVar = "oro_type.CDR-OAE" # name of variable
v='1' # Version of the script
dataFolder = '/homedata/dveytia/Product_1_data'
codedVariablesTxt = f'{dataFolder}/data/articleAnswers_formatted_2025-04-26.txt' # file with coded variables
unseenTxt = f'{dataFolder}/data/all_unseen_mitigation_oros.txt' # change to unique_references2.txt?
n_threads = 1 # number of threads to parallelize on
n_folds = 5 
rank_j = rank%n_folds # get rank
modType= 'multi-label' # type of model functions to load: either 'multi-label' or 'binary-label'
cv_results_fp = f'{dataFolder}/outputs/predictions/{targetVar}_predictions_eval_v{v}_k{rank_j}.csv'
model_weights_fp = f'{dataFolder}/outputs/model_weights'


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




################ Start defining functions ############################
tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(n_threads)

MODEL_NAME = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
   
with open(f'/home/dveytia/IPython_Notebooks/Product_1/pyFunctions/{modType}_1_predictions_functions.py') as f:
    exec(f.read())

outer_scores = []
clfs = []
    
####################### Target label to predict ##################################
def train_eval_bert(params, df, train, test, evaluate = True):
    train_dataset, val_dataset, MAX_LEN = create_train_val(df['text'].astype("str"), df[targetVar], train, test) #change here
    
    print("training bert with these params")
    print(params)
    model = init_model('distilbert-base-uncased', 1, params)
    model.fit(train_dataset.shuffle(100).batch(params['batch_size']),
              epochs=params['num_epochs'],
              batch_size=params['batch_size'],
              class_weight=params['class_weight']
    )

    preds = model.predict(val_dataset.batch(1)).logits
    y_pred = tf.keras.activations.sigmoid(tf.convert_to_tensor(preds)).numpy()
    if evaluate:
        eps = evaluate_preds(df[targetVar][test], y_pred[:,0]) #change here
        for key, value in params.items():
            eps[key] = value
        return eps, y_pred, model
    else:
        return y_pred, model


###### Reads in results from model selection and chooses the best model ######

outer_scores = []
inner_scores = []
params = ['batch_size','weight_decay','learning_rate','num_epochs','class_weight']

for k in range(3): 
    inner_df = pd.read_csv(f'{dataFolder}/outputs/model_selection/{targetVar}_model_selection_v{v}_k{k}.csv') 
    inner_df = inner_df.sort_values('F1',ascending=False).reset_index(drop=True)
    inner_scores += inner_df.to_dict('records')

inner_scores = pd.DataFrame.from_dict(inner_scores).fillna(-1)
best_model = (inner_scores
              .groupby(params)['F1']
              .mean()
              .sort_values(ascending=False)
              .reset_index() 
             ).to_dict('records')[0]


# can have a look at the F1 score for the best model
print(best_model)

del best_model['F1']
print(best_model)

if best_model['class_weight']==-1:
    best_model['class_weight']=None
else:
    best_model['class_weight'] = ast.literal_eval(best_model['class_weight'])



######################### Run model #######################################

outer_cv = KFold(n_splits=n_folds)
for k, (train, test) in enumerate(outer_cv.split(seen_index)):    
    if k!=rank_i:
        continue
    train = seen_index[train]
    test = unseen_index

    y_preds, model = train_eval_bert(best_model, df=df, train=train, test=test, evaluate=False)
    
    # Save results
    mw_fp = f'{model_weights_fp}/{targetVar}_{n_folds}fold/v{v}/k{k}'
    if not os.path.exists(mw_fp):
        os.makedirs(mw_fp)
        print(f'Weights directory created: {mw_fp}')
    else:
        print(f'Weights directory exists: {mw_fp}')
    model.save_pretrained(mw_fp, from_pt=True) 
    # # note to load the model: 
    # model = TFDistilBertForSequenceClassification.from_pretrained(mw_fp)

    np.save(f'{dataFolder}/outputs/predictions/{targetVar}_y_preds_{n_folds}fold_data_v{v}_k{k}.npz',y_preds)
    gc.collect()


np.save(f'{dataFolder}/outputs/predictions_data/{targetVar}_{n_folds}fold_data_pred_ids_v{v}.npz',df.loc[unseen_index,"id"]) # these are the unseen ids

print(t0 - time.time())
