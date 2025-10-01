import sys
print(sys.version)
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()



import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import ast
import time
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import tensorflow_addons as tfa
import os

t0 = time.time()


dataFolder = '/homedata/dveytia/Product_1_data'
targetVar = "oro_branch"
v = 'update'
modType= 'multi-label' 
n_folds = 5
rank_i = rank%n_folds
n_threads = 3
model_weights_fp = f'{dataFolder}/outputs/model_weights'



# Load seen documents
seen_df = pd.read_csv(f'{dataFolder}/data/all-coding-format-distilBERT-simplifiedMore.txt', delimiter='\t')
# seen_df = pd.read_csv('C:\\Users\\vcm20gly\\OneDrive - Bangor University\\Documents\\Review\\all-coding-format-distilBERT-simplifiedMore.txt', delimiter='\t')
seen_df = seen_df.rename(columns={'analysis_id':'id'})
seen_df['seen']=1

# Load unseen documents and merge
unseen_df = pd.read_csv(f'{dataFolder}/data/unique_references_UPDATE_13-05-2025.txt', delimiter='\t')
# unseen_df = pd.read_csv(r'C:/Users/vcm20gly/OneDrive - Bangor University/Documents/Review/0_unique_references.txt', delimiter='\t')
unseen_df = unseen_df.rename(columns={'analysis_id':'id'})
unseen_df=unseen_df.dropna(subset=['abstract']).reset_index(drop=True)

pred_df = pd.read_csv(f'{dataFolder}/outputs/predictions-compiled/oro_screen_update_predictions.csv')
# pred_df = pd.read_csv(r'C:/Users/vcm20gly/OneDrive - Bangor University/Documents/Review/03_Binary-AllText-NewApproach/1_document_relevance_v2.csv')

unseen_df = unseen_df.merge(pred_df, how="left")
unseen_df['seen']=0

# Choose which predictiction boundaries to apply
unseen_df = unseen_df[unseen_df['0 - relevance - mean_prediction']>=0.5]


# Concatenate seen and unseen
df = (pd.concat([seen_df,unseen_df])
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)


df["keywords"] = 'Keywords: '+ df["keywords"]
df["text"] = df[["title", "abstract", "keywords"]].agg(
    lambda x: ' '.join(x.dropna()), axis=1
)



seen_index = df[df['seen']==1].index
unseen_index = df[df['seen']==0].index

print("Dataset has been re-formatted and is ready")

# Start defining functions
tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(n_threads)

MODEL_NAME = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

with open(f'/home/dveytia/IPython_Notebooks/Product_1/pyFunctions/{modType}_1_predictions_functions.py') as f:
    exec(f.read())
    
   
def create_train_val(x,y,train,val):
    train_encodings = tokenizer(list(x[train].values),
                                truncation=True,
                                padding=True)
    val_encodings = tokenizer(list(x[val].values),
                                truncation=True,
                                padding=True) 
    
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        list(y[train].values)
    ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        list(y[val].values)
    ))
    
    
    MAX_LEN = train_dataset._structure[0]['input_ids'].shape[0]
    
    return train_dataset, val_dataset, MAX_LEN

def init_model(MODEL_NAME, num_labels, params):
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)  
    optimizer = tfa.optimizers.AdamW(learning_rate=params['learning_rate'], weight_decay=params['weight_decay'])

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model
    

####################### Select targets here ##################################
targets = [x for x in df.columns if targetVar in x]
df['labels'] = list(df[targets].values)

class_weight = {}
try:
    for i, t in enumerate(targets):
        cw = df[(df['random_sample']==1) & (df[t]==0)].shape[0] / df[(df['random_sample']==1) & (df[t]==1)].shape[0]
        class_weight[i] = cw
except:
    class_weight=None

outer_scores = []
clfs = []


def train_eval_bert(params, df, train, test, evaluate = True):
    train_dataset, val_dataset, MAX_LEN = create_train_val(df['text'], df['labels'], train, test)
    
    print("training bert with these params")
    print(params)
    model = init_model('distilbert-base-uncased', len(targets), params)
    model.fit(train_dataset.shuffle(100).batch(params['batch_size']),
              epochs=params['num_epochs'],
              batch_size=params['batch_size'],
              class_weight=params['class_weight']
    )

    preds = model.predict(val_dataset.batch(1)).logits
    y_pred = tf.keras.activations.sigmoid(tf.convert_to_tensor(preds)).numpy()
    ai = np.expand_dims(np.argmax(y_pred, axis=1), axis=1)
    maximums = np.maximum(y_pred.max(1),0.51)
    np.put_along_axis(y_pred, ai, maximums.reshape(ai.shape), axis=1)
    
    if evaluate:
        eps = evaluate_preds(df['relevant'][test], y_pred[:,0])  
        for key, value in params.items():
            eps[key] = value
        return eps, y_pred
    else:
        return y_pred

parallel=False

# Load best model (change file paths!)
outer_scores = []
inner_scores = []
params = ['batch_size','weight_decay','learning_rate','num_epochs','class_weight']


best_model = pd.read_csv(f'{dataFolder}/outputs/model_selection/ORO_sysMap_model_scores_2024-01-04.csv')
best_model = best_model.loc[11,params].to_dict()

if best_model['class_weight']=='-1':
    best_model['class_weight']=None
else:
    best_model['class_weight'] = ast.literal_eval(best_model['class_weight'])

print(best_model)


##################### Change paths x2 #####################################
outer_cv = KFold(n_splits=5)
for k, (train, test) in enumerate(outer_cv.split(seen_index)):    
    if k!=rank_i:
        continue
    train = seen_index[train]
    test = unseen_index

    # y_preds = train_eval_bert(best_model, df=df, train=train, test=test, evaluate=False)
    y_preds, model = train_eval_save_bert(best_model, df=df, train=train, test=test, evaluate=False)
    
    np.save(f"{dataFolder}/outputs/predictions/{targetVar}_{v}_y_preds_{n_folds}fold_{k}.npz",y_preds)
    
    # Save results
    mw_fp = f'{model_weights_fp}/{targetVar}_{n_folds}fold/v_{v}/k{k}'
    if not os.path.exists(mw_fp):
        os.makedirs(mw_fp)
        print(f'Weights directory created: {mw_fp}')
    else:
        print(f'Weights directory exists: {mw_fp}')
    model.save_pretrained(mw_fp, from_pt=True) 
    # # note to load the model: 
    # model = TFDistilBertForSequenceClassification.from_pretrained(mw_fp)


np.save(f"{dataFolder}/outputs/predictions_data/{targetVar}_{v}_unseen_ids.npz",df.loc[unseen_index,'id']) # these are the unseen ids

print(t0 - time.time())
