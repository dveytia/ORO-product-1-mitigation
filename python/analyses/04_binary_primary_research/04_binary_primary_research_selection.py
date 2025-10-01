import sys
print(sys.version)
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()


import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score
import itertools
import time

t0 = time.time()


"""
This version fits for all OROs together. Assumes that the features between primary and secondary research are consistent across the ORO types. I think this should be reasonable
"""

################# Change INPUTS ##################
targetVar = 'primary_research' # name of variable
v='1' # Version of the script
dataFolder = '/homedata/dveytia/Product_1_data'

codedVariablesTxt1 = f'{dataFolder}/data/articleAnswers_formatted_2025-03-17.txt'
codedVariablesTxt2 = f'{dataFolder}/data/articleAnswers_notMRE_formatted_2025-05-26.txt' # file with coded variables

n_threads = 5 # number of threads to parallelize on
n_folds = 3
rank_j = rank%n_folds
modType= 'binary-label' # type of model functions to load: either 'multi-label' or 'binary-label'
cv_results_fp = f'{dataFolder}/outputs/model_selection/{targetVar}_model_selection_v{v}_k{rank_j}.csv'



############################# Load data ###############################
######################## Change file paths x2 #########################
df1 = pd.read_csv(codedVariablesTxt1, delimiter='\t') 
df1 = df1.dropna(subset=["abstract"])

df2 = pd.read_csv(codedVariablesTxt2, delimiter='\t') 
df2 = df2.dropna(subset=["abstract"])

df = pd.concat([df1, df2]).drop_duplicates(subset='id')

# Keep only rows relevant for oro labels
oroCols = [x for x in df.columns if 'oro_type' in x]
df = df[df[oroCols].eq(1).any(axis=1)]

## cleaning -- check no relevant articles that aren't primary
# df.groupby('primary_research')['relevant'].sum()

# Sort values
df = (df
      #.query('unlabelled==0')
      # .query('relevant==1')
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)


# Concatenate all text together
df["keywords"] = 'Keywords: '+ df["keywords"]
df["text"] = df[["title", "abstract", "keywords"]].agg(
    lambda x: ' '.join(x.dropna()), axis=1
)

print("The data has been re-formatted")
print(df.shape)
df.primary_research.value_counts()

######################### Define functions #############################

tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(n_threads)

with open(f'/home/dveytia/IPython_Notebooks/Product_1/pyFunctions/{modType}_0_model-selection_functions.py') as f:
    exec(f.read())

MODEL_NAME = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


###################### calculate class weight here #################################

cw = df[(df['random_sample']==1) & (df[targetVar]==0)].shape[0] / df[(df['random_sample']==1) & (df[targetVar]==1)].shape[0]
class_weight={0:1, 1:cw}
class_weight

bert_params = {
  "class_weight": [None,class_weight],
  "batch_size": [16, 32], #
  "weight_decay": (0, 0.3),
  "learning_rate": (1e-5, 5e-5),
  "num_epochs": [2, 3, 4]
}


param_space = list(product_dict(**bert_params))

outer_cv = KFoldRandom(n_folds, df.index, df[df['random_sample']!=1].index, discard=False)


outer_scores = []
clfs = []


def train_eval_bert(params, df, train, test):
    train_dataset, val_dataset, MAX_LEN = create_train_val(df['text'].astype("str"), df[targetVar], train, test)
    
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
    eps = evaluate_preds(df[targetVar][test], y_pred[:,0])  
    for key, value in params.items():
        eps[key] = value
    return eps



############################## Run models ################################
######################## Change file path (x3) ###########################
for k, (train, test) in enumerate(outer_cv):    
    if k!=rank_j:
        continue
    try:
        pr = param_space[0]
        cv_results=pd.read_csv(cv_results_fp).to_dict('records') 
        params_tested=pd.read_csv(cv_results_fp)[list(pr.keys())].to_dict('records')
    except:
        cv_results = []
        params_tested = []
    for pr in param_space:
        if pr in params_tested:
            continue
        cv_results.append(train_eval_bert(pr, df=df, train=train, test=test))
        pd.DataFrame.from_dict(cv_results).to_csv(cv_results_fp,index=False) 
        gc.collect()
