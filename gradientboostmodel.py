import helpers
import numpy as np
from regression_tree import build_tree, predict_tree
class gradient_boost_model:
    forest=[]
    baseline=0
    weight=0
    maxheight=0
    def __init__(self,lr,maxheight=2):
        self.weight=lr
        self.maxheight=maxheight
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
            print(i)
            print(helpers.mse(res))
            i+=1
    def predict_many(self,X):
        return helpers.predict_all(X,self.baseline,self.forest,self.weight)
    def predict_one(self,x):
        return helpers.predict_one(x,self.baseline,self.forest,self.weight)