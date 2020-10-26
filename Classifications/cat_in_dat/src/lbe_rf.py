# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 23:54:36 2020

@author: samu0315
"""

import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    # load the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    
    # all columns are features except id,target and kfold
    features = [ f for f in df.columns if f not in ("id","target","kfold")]
    
    # fill all the missing values with NONE
    # note: we are converting all columns to strings
    # its ok as we are looking at ways to handle the categorical data
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")
        
    # now its time to label encode the features
    for col in features:
        # initialize LabelEncoder for each feature column
        lbl = preprocessing.LabelEncoder()
        
        # fit label encoder on all data
        lbl.fit(df[col])
        
        # transform all the data
        df.loc[:, col] = lbl.transform(df[col])
    
    # get the trainign data for this fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # get validation data
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # get train_data
    x_train = df_train[features].values
    
    # get test data
    x_valid = df_valid[features].values
    
    # initialize logistc model
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    
    # fit model on train ohe
    model.fit(x_train,df_train.target.values)
    
    # Predict on validation data
    # we are using AUC as metrci
    # hence we need prob as output
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # get auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    
    # Print AUC
    print(f"Fold = {fold}, AUC = {auc}")
    
if __name__ == "__main__":
    for fold in range(5):
        run(fold)