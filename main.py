import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from regression_tree import build_tree, predict_tree
import numpy as np
from gradientboostmodel import gradient_boost_model
import helpers

def main():
    print("Comparing various model implementations...")
    MSE_comparison()
    print("Done")
    print("Creating MSE plot...")
    create_MSE_plot()
    print("Done")
    print("Testing a range of hyperparameter values...")
    test_over_range_of_hyp()
    print("Done")

def test_over_range_of_hyp(): #Tests over a small range of hyperparameters
    X = pd.read_csv("data/Training_cleaned_features.csv")
    y = pd.read_csv("data/Training_cleaned_target.csv")["charges"]
    X_val=pd.read_csv("data/Validation_cleaned_features.csv")
    y_val = pd.read_csv("data/Validation_cleaned_target.csv")["charges"]
    treeheights=[2,3]
    learningrates=[0.001,0.0025,0.005,0.0075,0.01]
    results=[[None]+learningrates]
    for tree_depth in treeheights:
        results.append([tree_depth])
        for lr in learningrates:
            #print(tree_depth,lr)
            ours=gradient_boost_model(lr,tree_depth)
            ours.train(X,y,2500)
            oursval=ours.predict_many(X_val)
            oursres=helpers.residuals(np.asarray(y_val),np.asarray(oursval))
            #print(helpers.mse(oursres))
            results[-1].append(helpers.mse(oursres))     
    out=pd.DataFrame(results)
    out.to_csv("results_smaller.csv", index=False)
def create_MSE_plot(depth=2,lr=0.0025):
    X = pd.read_csv("data/Training_cleaned_features.csv")
    y = pd.read_csv("data/Training_cleaned_target.csv")["charges"]
    X_val=pd.read_csv("data/Validation_cleaned_features.csv")
    y_val = pd.read_csv("data/Validation_cleaned_target.csv")["charges"]
    model=gradient_boost_model(lr,depth)
    model.trainandlogmse(X,y,X_val,y_val,3000)

def MSE_comparison(depth=2,lr=0.0025):
    X = pd.read_csv("data/Training_cleaned_features.csv")
    y = pd.read_csv("data/Training_cleaned_target.csv")["charges"]
    X_val=pd.read_csv("data/Validation_cleaned_features.csv")
    y_val = pd.read_csv("data/Validation_cleaned_target.csv")["charges"]
    model=gradient_boost_model(lr,depth)
    model.train(X,y,2500)
    oursval=model.predict_many(X_val)
    oursres=helpers.residuals(np.asarray(y_val),np.asarray(oursval))
    print("Validation MSE of our model:")
    print(helpers.mse(oursres))
    basereg=GradientBoostingRegressor(loss="squared_error")
    basereg.fit(X,y)
    baseval=basereg.predict(X_val)
    baseres=helpers.residuals(np.asarray(y_val),np.asarray(baseval))
    print("Validation MSE of base Scikit-learn model:")
    print(helpers.mse(baseres))
    customreg=GradientBoostingRegressor(loss="squared_error",n_estimators=2500,learning_rate=0.0025,max_depth=2)
    customreg.fit(X,y)
    customval=customreg.predict(X_val)
    customres=helpers.residuals(np.asarray(y_val),np.asarray(customval))
    print("Validation MSE of Scikit-learn model with our calculated optimal hyperparamters:")
    print(helpers.mse(customres))

main()
