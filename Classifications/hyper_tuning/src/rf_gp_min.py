# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 23:17:07 2020

@author: samu0315
"""

# rf_gp_minimise.py
import numpy as np
import pandas as pd

from functools import partial

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from skopt import gp_minimize
from skopt import space

def optimize(params, param_names, x, y):
    """
    This is the main optimization function.
    This function takes arguments from search space and training 
    features and targets.It then stablizes the models
    by setting the chosen params and runs cv and returns -ve accuracy scores.
    :params : list of params from gp_minimize
    :param_names :list of names for parameters and preserve the order
    :x : training data
    :y : labels
    :return : negative accuracy
    """
    # convert params to dict
    params = dict(zip(param_names,params))
    
    # initialize model with current parameters
    model = ensemble.RandomForestClassifier(**params)
    
    # initialize kfold stratified as this is classification setting
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # initialse a list to store accuracy values
    accuracies = []
    
    # loop over all folds
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        
        x_train = x[train_idx]
        y_train = y[train_idx]
        
        x_test = x[test_idx]
        y_test = y[test_idx]
        
        # fit model to current fold
        model.fit(x_train,y_train)
        
        # create preds
        preds = model.predict(x_test)
        
        # calc the acuracy socre
        fold_accuracy = metrics.accuracy_score(
                y_test,
                preds
                )
        accuracies.append(fold_accuracy)
        
    # return -ve of accuracy
    return  -1 * np.mean(accuracies)
    
if __name__ == "__main__":
	# read the data
	df = pd.read_csv('../input/train.csv')
	
	# get X and y
	X = df.drop('price_range',axis=1).values
	y = df.price_range.values
	
	# define param space
	param_space = [
			# max depth is an int value 
			space.Integer(3,15,name = "max_depth"),
			# n_estimators is int 
			space.Integer(100,1500,name = "n_estimators"),
			# criterion is a category. here we define list of categories
			space.Categorical(["gini", "entropy"], name="criterion"),
			# you can also have Real numbered space and define a
			# distribution you want to pick it from
			space.Real(0.01, 1, prior="uniform", name="max_features")
			]
	# make a list of param names
	# this has to be same order as the search space
	# inside the main function
	param_names = [
			"max_depth",
			"n_estimators",
			"criterion",
			"max_features"
			]
	
	# *************TRICKY PART HERE******************
	# gp_minimize expects function to have only one parameter
	# to adhere to that, by using functools partial, i am creating a
	# new function which has same parameters as the
	# optimize function except for the fact that
	# only one param, i.e. the "params" parameter is
	# required. this is how gp_minimize expects the
	# optimization function to be. you can get rid of this
	# by reading data inside the optimize function or by
	# defining the optimize function here.
	optimization_function = partial(
			optimize,
			param_names = param_names,
			x= X,
			y= y
			)
	
	# now we are all set to optimization
	# we wil use skopt, gp_optimize which uses bayesian optimization
	result = gp_minimize(
			optimization_function,
			dimensions = param_space,
			n_calls = 40, # number of evaluations
			n_random_starts = 10,
			verbose =10
			)
	
	# create best params dict and print
	best_params = dict(
			zip(
					param_names,
					result.x
				)
			)
	print(best_params)