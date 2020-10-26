# create_folds.py
# this code will create folds of the src data

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    
    # Read Training data
    df = pd.read_csv(config.INPUT_FILE)
    
    # we create a new column called kfold and fill with -1
    df['kfold'] = -1
    
    # the next step is to shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # fetch labels
    y = df.target.values
    
    # call the kfold class from model selection
    kf = model_selection.StratifiedKFold(n_splits = 5)
    
    # fill the new kfold column
    for fold , (train_index,valid_index) in enumerate(kf.split(X=df,y=y)):
        df.loc[valid_index,'kfold'] = fold
        
    # Save the new csv with kfold column
    df.to_csv(config.TRAINING_FILE, index=False)