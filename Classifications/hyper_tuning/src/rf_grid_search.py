# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 22:19:06 2020

@author: samu0315
"""

# rf_grid_search.py
import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":
    # read the train data
    df = pd.read_csv('../input/train.csv')
    
    # Features are all columns except price range
    # Make X and y from df
    X = df.drop('price_range',axis=1).values
    y = df.price_range.values
    
    # define the model here, use RF
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    
    # define params grid
    param_grid = {
            "n_estimators":[100,200,500,800,1000],
            "max_depth":[1,3,5,7,9,11],
            "criterion":["gini","entropy"]
            }
    
    # initialise grid search 
    model = model_selection.GridSearchCV(
            estimator = classifier,
            param_grid = param_grid,
            scoring = "accuracy",
            verbose = 10,
            n_jobs = -1,
            cv = 5
            )
    
    # fit model and extract best score
    model.fit(X,y)
    print(f"Best Score:{model.best_score_}")
    
    print("Best Parameters set")
    # get best params
    best_params = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}:{best_params[param_name]}")
    