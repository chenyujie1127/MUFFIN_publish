
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,average_precision_score


def accuracy(y_true, y_pred):
    
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred,multi_type=True):
    
    # binary: whether the task is binary-class or not
    
    if not multi_type:
        return precision_score(y_true, y_pred)
    else:
        return precision_score(y_true, y_pred, average='macro')


def recall(y_true, y_pred,multi_type=True):
    
    # binary: whether the task is binary-class or not
    
    if not multi_type:
        return recall_score(y_true, y_pred)
    else:
        return recall_score(y_true, y_pred, average='macro')


def f1(y_true, y_pred,multi_type=True):
    
    # binary: whether the task is binary-class or not
    
    if not multi_type:
        return f1_score(y_true, y_pred)
    else:
        return f1_score(y_true, y_pred, average='macro')


def auc(y_true,y_pred_score):
    
    # for multi-class / multi-label task 

    return roc_auc_score(y_true, y_pred_score, average='macro')

def aupr(y_true,y_pred_score):

    # for multi-class / multi-label task 

    return average_precision_score(y_true,y_pred_score, average="macro")