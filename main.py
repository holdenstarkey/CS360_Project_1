import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

def main():
    training=pd.read_csv("/homes/hstarkey/CS360_Project_1/data/Training_Validation_Testing Insurance CS360 - Training.csv")
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