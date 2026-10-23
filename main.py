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
    
main()