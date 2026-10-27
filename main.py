import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from regression_tree import build_tree, predict_tree
import numpy as np

# as arrays
X = pd.read_csv("data/Training_cleaned_features.csv").to_numpy(dtype=float)
y = pd.read_csv("data/Training_cleaned_target.csv")["charges"].to_numpy(dtype=float)

# Build the weak regression tree at depth 2
tree = build_tree(X, y, max_depth=2)
#print(tree)

# Predict charges for each row
predictions = np.array([predict_tree(tree, row) for row in X])

#print(predictions)
def main():
    X = pd.read_csv("data/Training_cleaned_features.csv")
    y = pd.read_csv("data/Training_cleaned_target.csv")["charges"]
    X_val=pd.read_csv("data/Validation_cleaned_features.csv")
    y_val = pd.read_csv("data/Validation_cleaned_target.csv")["charges"]
    results=[]
    for tree_depth in [2,3,4]:
        results.append([])
        for lr in [0.05,0.1,0.2,0.25,0.3,0.5,0.6,0.75,1,3,5]:
            model=train(X,y,2500,lr,tree_depth)
            predictions=model.predict_many(X_val)
            res=residuals(np.asarray(y_val),np.asarray(predictions))
            results[-1].append(mse(res))
    out=pd.DataFrame(results)
    out.to_csv("results.csv", index=False)




    
def split_targets(dataframe):
    targets=dataframe["charges"]
    data=dataframe.drop(["charges"])
    return (data,targets)

def correct_text_labels(dataframe):
    dataframe['sex']=pd.factorize(dataframe['sex'])[0]
    dataframe['smoker']=pd.factorize(dataframe['smoker'])[0]
    dataframe['region']=pd.factorize(dataframe['region'])[0]
    return dataframe
def sklearn_implementation(X,y):
    gb_rg=GradientBoostingRegressor(n_estimators=20,learning_rate=0.05)
    gb_rg.fit(X,y)
    return gb_rg
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

def train(X,y,iterations,lr,md=2):
    init_pred=baseline(y)
    init_preds=populate_list(init_pred,len(y))
    res=residuals(np.asarray(y),init_preds)
    trees=[]
    treepreds=[]
    i=0
    while i<iterations:
        trees.append(build_tree(np.asarray(X),np.asarray(res),max_depth=md))
        treepreds.append(np.array([predict_tree(trees[-1],x) for x in np.asarray(X)]))
        predictions=np.asarray(init_preds)+lr*sum(treepreds)
        res=residuals(np.asarray(y),predictions)
        #print(i)
        #print(mse(res))
        i+=1
    return gradient_boost_model(init_pred,trees,lr)

class gradient_boost_model:
    forest=[]
    baseline=0
    weight=0

    def __init__(self,baseline,trees,lr):
        self.baseline=baseline
        self.forest=trees
        self.weight=lr
        pass
    def predict_many(self,X):
        return predict_all(X,self.baseline,self.forest,self.weight)
    def predict_one(self,x):
        return predict_one(x,self.baseline,self.forest,self.weight)


main()
