# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:51:51 2020

@author: samu0315
"""
# lbl_xgb_num.py

import pandas as pd
import xgboost as xgb

import config

from sklearn import preprocessing
from sklearn import metrics

def run(fold):
    # load the input data
    df = pd.read_csv(config.TRAINING_FILE)
    
    # list the numerical columns
    num_cols = [
            "fnlwgt",
            "age",
            "capital-gain",
            "capital-loss",
            "hours-per-week"
            ]
    
    # drop numerical columns
    # df = df.drop(num_cols,axis = 1)
    
    # map targets to 0 and 1
    target_mapping = {
            "<=50K":0,
            ">50K":1
           }
    df.loc[:,"income"] = df.income.map(target_mapping)
    
    # all columns are features now
    # except kfold and income
    features = [
            f for f in df.columns if f not in ("kfold","income")
            ]
    
    # fill all missing values with NONE 
    # XGB being a tree based model, this works
    for col in features :
        if col not in num_cols:
            df.loc[:,col] = df[col].astype(str).fillna("NONE")
        
    # Label encode the coulmns
    for col in features:
        # initialise label encoder
        le = preprocessing.LabelEncoder()
        
        # fit on the full column data
        le.fit(df[col])
        
        # transform the data
        df.loc[:,col] = le.transform(df[col])
    
    # get the training data
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # get the test data
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # get X_train
    x_train = df_train[features].values
    
    # get testing x
    x_valid = df_valid[features].values
    
    # initilize XGBoost model
    model = xgb.XGBClassifier(
            n_jobs=-1,
            max_depth = 7,
            n_estimators = 200
            )
    
    # fit the model on training data
    model.fit(x_train,df_train.income.values)
    
    # predict on validation data
    # we are considering the AUC as metric
    valid_preds = model.predict_proba(x_valid)[:,1]
    
    # get auc socre
    auc = metrics.roc_auc_score(df_valid.income.values,valid_preds)
    
    # print to screen
    print(f"Fold = {fold} & AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)