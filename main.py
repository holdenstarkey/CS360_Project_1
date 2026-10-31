import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from regression_tree import build_tree, predict_tree
import numpy as np
from gradientboostmodel import gradient_boost_model
import helpers

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
    # for tree_depth in [2,3,4]:
    #     results.append([])
    #     for lr in [0.05,0.1,0.2,0.25,0.3,0.5,0.6,0.75,1]:
    #         model=train(X,y,2500,lr,tree_depth)
    #         predictions=model.predict_many(X_val)
    #         res=residuals(np.asarray(y_val),np.asarray(predictions))
    #         results[-1].append(mse(res))
    skresults=[]
    '''
    for tree_depth in [2,3,4]:
        results.append([])
        skresults.append([])
        for lr in [0.01,0.02,0.03,0.04]:
            #model=train(X,y,2500,lr,tree_depth)
            #predictions=model.predict_many(X_val)
            #res=residuals(np.asarray(y_val),np.asarray(predictions))
            #results[-1].append(mse(res))
            reg=GradientBoostingRegressor(loss="squared_error")
            reg.fit(X,y)
            skval=reg.predict(X_val)
            skres=residuals(np.asarray(y_val),np.asarray(skval))
            skresults[-1].append(mse(skres))
    '''
    ours=gradient_boost_model(0.01,2)
    ours.train(X,y,2500)
    oursval=ours.predict_many(X_val)
    oursres=helpers.residuals(np.asarray(y_val),np.asarray(oursval))
    print(helpers.mse(oursres))
    reg=GradientBoostingRegressor(loss="squared_error")
    reg.fit(X,y)
    skval=reg.predict(X_val)
    skres=helpers.residuals(np.asarray(y_val),np.asarray(skval))
    print(helpers.mse(skres))
    #skout=pd.DataFrame(skresults)
    #skout.to_csv("skresults_smaller.csv", index=False)       
    #out=pd.DataFrame(results)
    #out.to_csv("results_smaller.csv", index=False)
    # out.to_csv("results.csv", index=False)




    




main()
