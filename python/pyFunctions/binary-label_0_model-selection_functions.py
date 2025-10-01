## FUNCTIONS FOR BINARY LABELS MODEL SELECTION

def KFoldRandom(n_splits, X, no_test, shuffle=False, discard=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    for train, test in kf.split(X):
        if not discard:
            train = list(train) +  [x for x in test if x in no_test]
        test = [x for x in test if x not in no_test]
        yield (train, test)


def KFoldRandomModified(n_splits, X, no_test, no_train, shuffle=False, discard=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    for train, test in kf.split(X):
        if not discard:
            train = list(train) +  [x for x in test if x in no_test]
            test = list(test) + [x for x in test if x in no_train]
        test = [x for x in test if x not in no_test]
        train = [x for x in train if x not in no_train]
        yield (train, test)

from sklearn.model_selection import KFold
import numpy as np


class KFoldWithFixedTest:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, random_sample=None):
        if random_sample is None:
            raise ValueError("You must provide the `random_sample` array.")

        X = np.array(X)
        random_sample = np.array(random_sample)

        # Identify indices
        fixed_test_idx = np.where(random_sample == 1)[0]
        train_candidates_idx = np.where(random_sample == 0)[0]

        # Apply KFold only to training candidates
        for train_idx_raw, _ in self.kf.split(train_candidates_idx):
            train_idx = train_candidates_idx[train_idx_raw]
            test_idx = fixed_test_idx
            yield train_idx, test_idx




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

def evaluate_preds(y_true, y_pred):
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except:
        roc_auc = np.NaN
    f1 = f1_score(y_true, y_pred.round())
    p, r = precision_score(y_true, y_pred.round()), recall_score(y_true, y_pred.round())
    acc = accuracy_score(y_true, y_pred.round())
    print(f"ROC AUC: {roc_auc:.0%}, F1: {f1:.1%}, precision: {p:.1%}, recall {r:.1%}, acc {acc:.0%}")
    return {"ROC AUC": roc_auc, "F1": f1, "precision": p, "recall": r, "accuracy": acc}



def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

