# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:29:39 2020

@author: samu0315
"""

# rf_hyperopt.py

import numpy as np
import pandas as pd

from functools import partial

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from hyperopt import hp, fmin, tpe, Trails

from hyperopt.pll.base import scope

def optimize(params, x, y):
    """
    This is main optimization function
    Takes all params from seach space
    and data from input,
    Fits a model on kfold split data
    return -ve accuracy score to optimize
    """
    # initilize model with current params
    model = ensemble.RandomForestClassifiers(**params)
    
    # initialize stratified kfold
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # empty list to store accuracies
    accuracies = []
    
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        
        x_train = x[train_idx]
        y_train = y[train_idx]
        
        x_test = x[test_idx]
        y_test = y[test_idx]
        
        # fit model to current fold
        model.fit(x_train,y_train)
        
        # get predictions
        preds = model.predict(x_test)
        
        # get accuracy score
        acc = metrics.accuracy_score(
                y_test,
                preds
                )
        accuracies.append(acc)
        
    # return -ve mean of all folds
    return -1 * np.mean(accuracies)

if __name__=="__main__":
    # read the training data
    df = pd.read_csv("../input/train.csv")
    
    # features are all columns without price_range
    # note that there is no id column in this dataset
    # here we have training features
    X = df.drop("price_range", axis=1).values
    
    # and the targets
    y = df.price_range.values
    
    # define parameter space
    # now we use hyperopt
    param_space={
            # quiform gives round(uniform(min,max)/q)*q
            # for trees depth and estimators we want to use int
            "max_depth" : scope.int(hp.quniform("max_depth",1,15,1)),
            "n_estimators" : scope.int(hp.quniform("n_estimators",100,1500,1)),
            # choice chooses from a list
            # can be used for categorical
            "criterion" : hp.choice("criterion",["gini","entropy"]),
            # uniform chooses a value between 2 values
            "max_features" : hp.uniform("max_features",0,1)
            }
    
    # partial function
    optimization_function = 
        