# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 23:17:29 2020

@author: samu0315
"""

import pandas as pd

from sklearn import linear_model
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
        
    # get the trainign data for this fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # get validation data
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # get OHE from sklearn
    ohe = preprocessing.OneHotEncoder()
    
    #fit ohe on train+valid dfs
    full_data = pd.concat(
            [df_train[features],df_valid[features]]
            ,axis=0
            )
    ohe.fit(full_data[features])
    
    # transform training data
    x_train = ohe.transform(df_train[features])
    
    # transfrom validation data
    x_valid = ohe.transform(df_valid[features])
    
    # initialize logistc model
    model = linear_model.LogisticRegression()
    
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