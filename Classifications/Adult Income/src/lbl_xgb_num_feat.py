# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:56:24 2020

@author: samu0315
"""
#lbl_xgb_num_feat.py

import pandas as pd
import xgboost as xgb
import itertools

import config

from sklearn import preprocessing
from sklearn import metrics


def feature_engineering(df, cat_cols):
    """
    This function is used for feature engineering
    :param df: the pandas dataframe with train/test data
    :param cat_cols: list of categorical columns
    :return: dataframe with new features
    """
    # this will create all 2-combinations of values
    # in this list
    # for example:
    # list(itertools.combinations([1,2,3], 2)) will return
    # [(1, 2), (1, 3), (2, 3)]
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[
            :,
            c1 + "_" + c2
        ] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df

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
    
    # map targets to 0 and 1
    target_mapping = {
            "<=50K":0,
            ">50K":1
           }
    df.loc[:,"income"] = df.income.map(target_mapping)

    # list of categorical columns for feature engineering
    cat_cols = [
            c for c in df.columns if c not in num_cols
            and c not in ("kfold", "income")
            ]
    # add new features
    df = feature_engineering(df, cat_cols)
    
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