import numpy as np
from regression_tree import build_tree, predict_tree

def residuals(y_true:np.ndarray,y_pred:np.ndarray):
    if len(y_true)!=len(y_pred):
            return ValueError("Input lengths don't match")
    return y_true-y_pred
def mse(residuals):
    return sum(residuals*residuals)/len(residuals)
def baseline(y_true):
    return sum(y_true)/len(y_true)
def populate_list(value,length):
    return np.asarray([value for _ in range(length)])

def predict_one(x,baseline,trees,lr):
    treepreds=[]
    for tree in trees:
        treepreds.append(predict_tree(tree,x))
    return baseline+lr*sum(np.asarray(treepreds))

def predict_all(X,baseline,trees,lr):
    preds=[]
    for _,i in X.iterrows():
        preds.append(predict_one(i,baseline,trees,lr))
    return preds