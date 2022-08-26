import config
import pandas as pd 
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    df = pd.read_csv(config.TRAINING_FILE)

    df = df.sample(frac=1).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=5)
    
    for fold, (train_, cv_) in enumerate(skf.split(X=df, y=df['Loan_Status'])):
        df.loc[cv_, 'kfold'] = fold
    
    df.to_csv('../input/train_folds.csv', index=False)