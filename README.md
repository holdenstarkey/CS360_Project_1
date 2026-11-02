# CS360 Project 1

Rohan Sarma, Jersson Veliz, Holden Starkey

## What was implemented:

Gradient boosting for regression, used to predict medical costs (target) for an individual given features of their personal background. This process was mainly broken down into three parts: data processing, tree implementation, and model training.

Training the model involves combining many of these weak tree predictors into a single model, such that their combination results in a single strong model. The nth tree tries to predict the residuals of the model up to the n-1th tree, meaning that each iteration of the model slightly corrects for the errors in the previous model. By doing this, we can form a single strong predictor out of many weak predictors

## Libraries used:

- Pandas, used for processing and sorting the dataset
- Numpy, used for its efficient numerical processing capabilities (namely for nparray arithmetic)
- Scikit-learn, compared to its built-in gradient boosting (for regression) methods to evaluate the results of our model
- Matplotlib, used for displaying final comparison figure for baseline mean vs our model vs scikit-learn model

## Supported features/parameters:

### In helpers.py:

- Basic data science method implementations: ***residuals***(ytrue-ypred), ***mse***(residuals), ***baseline***(y_true) [returns mean value of targets], ***populate_list***(value,length) [returns an array with one value repeated for a given length]

- Predictors for a gradient boosting model: ***predict_one***(x,baseline,trees,lr) [predicts the output of a single set of features x, given a model’s baseline, trees, and learning rate], ***predict_all***(X,baseline,trees,lr) [predicts outputs for all entries of a dataset]

In **gradientboostmodel.py** we’ve created our model class, with the functions: ***\_\_init\_\_***(self,lr,maxheight) [creates an empty model with a given learning rate/weight and tree height], ***train***(self,X,y,iterations) [trains the model on the provided training data for the specified number of iterations, calculating a baseline and creating a prediction forest], ***trainandlogmse***(self,X,y,X_val,y_val,iterations) [trains a model in the same way as the previous function, but also creates a graph of the training and validation MSE over the course of the training process. Saves this graph as MSE_over_training.png.], _**predict_many**_**(**self,X) [predicts targets given a dataset], ***predict_one***(self,x) [predicts a single target given a set of features]

In **regression_tree.py** we’ve created the functions: ***best_split***(X,y) [finds the feature and threshold that minimize the total squared error across the two resulting groups], ***build_tree***(X,y,depth=0,max_depth=2)[[recursively creates a regression tree up to the specified maximum depth, using the mean target value as the prediction at each leaf], ***predict_tree***(tree,x)[predicts a single target by following the tree’s feature thresholds until reaching a leaf].

In **final_testing_run.py** we’ve created the functions: ***main()***: [loads the training and testing data, cleans and encodes the testing features to match the training data, and trains our model and sklearn’s gradient boosting model using the same learning rate, tree depth, and number of iterations. Compares both models against a mean baseline using MSE, RMSE, R2, prints and saves the results as final_test_results.csv, and saves an RMSE comparison graph as final_test_comparison.png], ***get_metrics***(name,predictions) [a helper function inside main() that calculates and returns a model’s name, MSE, RMSE, and R2 on the testing data.]

## Datasets and Preprocessing:

[Medical Cost Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance?resource=download&select=insurance.csv)

70% training, 15% validation, 15% testing

- Hot-one encoded the categorical features of the dataset(e.g. region)
- Turned the original 6 features into 11 features
- Kept testing data separate to prevent leakage into our algorithm while training

## How to reproduce our results:

Run the **main()** function in our **main.py** file, which should produce:

- **results_smaller.csv**, a csv containing the validation MSE values for a model trained on various hyperparameter values over 2500 iterations. This file was used to determine which hyperparameter combination results in the lowest loss for validation data. (We also tested combinations outside of this range, this output was shortened for runtime/simplicity when recreating.)

- ***MSE_over_training.png***, an image showing training and validation MSE over the span of one 3000 iteration training run. Validation MSE was calculated every 50 iterations to save time.

- A terminal output comparing the validation MSE of our model with optimal hyperparameters against the validation MSE of the base Scikit-learn gradient-boosting implementation and the validation MSE of the Scikit-learn gradient-boosting implementation with our optimal hyperparameters.

## Results:

Based on the performance of various hyperparameter testing, we selected a tree depth of 2 and a learning rate of 0.0025 as our hyperparameters (as they resulted in the lowest validation MSE). Additionally, we chose to use 2500 iterations as our training amount, as validation MSE appears to have leveled out to a minimum at that point (see **MSE_over_training.png**). Comparing our implementation to a comparable Scikit-learn gradient boosting implementation, we receive near identical results across all metrics.
