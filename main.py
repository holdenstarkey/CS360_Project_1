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
    training=pd.read_csv("data/Training_Validation_Testing Insurance CS360 - Training.csv")
    #Preprocess data here
    training_converted=correct_text_labels(training)
    y=training_converted["charges"]
    X=training_converted.drop(columns="charges")
    print(train(X,y,10,.1))
    
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
def residuals(y_true,y_pred):
    if len(y_true)!=len(y_pred):
            return ValueError("Input lengths don't match")
    res=[]
    for i in range(len(y_true)):
        res.append(y_true[i]-y_pred[i])
    return res
def mse(residuals):
    sum=0
    for i in residuals:
        sum+=(residuals)**2
    sum/=len(residuals)
    return sum
def baseline(y_true):
    return sum(y_true)/len(y_true)
def populate_list(value,length):
    return [value for _ in range(length)]

def predict_one(x,baseline,trees,lr):
    treepreds=[]
    for tree in trees:
        treepreds.append(predict_tree(tree,x))
    finalpred=baseline
    for i in treepreds:
        finalpred+=lr*i
    return finalpred

def predict_all(X,baseline,trees,lr):
    preds=[]
    for _,i in X.iterrows():
        preds.append(predict_one(i,baseline,trees,lr))
    return preds

def train(X,y,iterations,lr):
    init_pred=baseline(y)
    init_preds=populate_list(init_pred,len(y))
    res=residuals(y,init_preds)
    trees=[]
    i=0
    while i<iterations:
        #Error occurs for this line, fitting tree to residuals
        trees.append(build_tree(X,res))
        
        predictions=predict_all(X,init_pred,trees,lr)
        res=residuals(y,predictions)
        i+=1
    return baseline,trees
    
    



main()
