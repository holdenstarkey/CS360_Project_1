import helpers
import numpy as np
from regression_tree import build_tree, predict_tree
import matplotlib.pyplot as plt
class gradient_boost_model:
    def __init__(self,lr,maxheight=2):
        self.weight=lr
        self.maxheight=maxheight
        self.forest=[]
        self.baseline=0
        pass
    def train(self,X,y,iterations):
        self.baseline=helpers.baseline(y)
        init_preds=helpers.populate_list(self.baseline,len(y))
        res=helpers.residuals(np.asarray(y),init_preds)
        treepreds=[]
        i=0
        while i<iterations:
            self.forest.append(build_tree(np.asarray(X),np.asarray(res),max_depth=self.maxheight))
            treepreds.append(np.array([predict_tree(self.forest[-1],x) for x in np.asarray(X)]))
            predictions=np.asarray(init_preds)+self.weight*sum(treepreds)
            res=helpers.residuals(np.asarray(y),predictions)
            #print(i)
            #print(helpers.mse(res))
            i+=1
    def trainandlogmse(self,X,y,X_val,y_val,iterations):
        self.baseline=helpers.baseline(y)
        init_preds=helpers.populate_list(self.baseline,len(y))
        res=helpers.residuals(np.asarray(y),init_preds)
        treepreds=[]
        trainmses=[]
        valmses=[]
        i=0
        while i<iterations:
            self.forest.append(build_tree(np.asarray(X),np.asarray(res),max_depth=self.maxheight))
            treepreds.append(np.array([predict_tree(self.forest[-1],x) for x in np.asarray(X)]))
            predictions=np.asarray(init_preds)+self.weight*sum(treepreds)
            res=helpers.residuals(np.asarray(y),predictions)
            trainmses.append(helpers.mse(res))
            if i%50==0:
                validation_predictions=self.predict_many(X_val)
                validation_residuals=helpers.residuals(np.asarray(y_val),validation_predictions)
                valmses.append(helpers.mse(validation_residuals))
            #print(i)
            #print(helpers.mse(res))
            i+=1
        plt.plot(range(len(trainmses)),trainmses,label="training values",color='b')
        plt.plot([50*i for i in range(len(valmses))],valmses,label="validation values",color='r')
        plt.legend()
        plt.xlabel("Iteration #")
        plt.ylabel("MSE")
        plt.title("MSE over training iterations for training/validation datasets")
        plt.savefig("MSE_over_training.png")
    def predict_many(self,X):
        return helpers.predict_all(X,self.baseline,self.forest,self.weight)
    def predict_one(self,x):
        return helpers.predict_one(x,self.baseline,self.forest,self.weight)