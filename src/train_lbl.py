import config
import model_dispatcher

import pandas as pd 
import numpy as np 
import itertools
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

def num_preprocessing(df, num_cols):
    pf = PolynomialFeatures(
        degree=2,
        interaction_only=False,
        include_bias=False
    )

    pf.fit(df[num_cols])
    poly_feats = pf.transform(df[num_cols])

    num_feats = poly_feats.shape[1]

    new_df = pd.DataFrame(
        poly_feats,
        columns=[f'f_{i}' for i in range(1, num_feats + 1)]
    )

    df = df.join(new_df)
    return df

def feature_engineering(df, cat_cols):

    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['IncomePerMember'] = df['TotalIncome']/df['Dependents']
    df['EMI'] = (df['LoanAmount']*2000) / df['Loan_Amount_Term']
    df['EMI_Income'] = df['EMI'] / df['TotalIncome']
    df['CanPay'] = df['EMI_Income'].apply(lambda i: 1 if i<=0.4 else 0)
    df['IncomePerLoan'] = df['TotalIncome'] / df['LoanAmount']
    df['AreaWise'] = df['Property_Area'].apply(lambda i: 0.38 if i=='Rural' else (0.7 if i=='Semiurban' else 0.48))

    """
    comb_cols = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in comb_cols:
        df.loc[:, c1 + '_' + c2] = df[c1].astype(str) + '_' + df[c2].astype(str)
    """

    return df 

def test_predictions(clf, cat_cols, dependent_mapping):
    
    test_df = pd.read_csv(config.TEST_FILE)
   
    test_df['Dependents'] = test_df['Dependents'].map(dependent_mapping)
    test_df['Dependents'] = test_df['Dependents'] + 1
    
    num_features = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Dependents'
    ]

    test_df[num_features] = test_df[num_features].fillna(test_df[num_features].median())
    test_df['Credit_History'] = test_df['Credit_History'].fillna(1)
    
    test_df = feature_engineering(test_df, cat_cols)

    num_features = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'EMI',
        'TotalIncome',
        'IncomePerMember'
    ]

    #test_df = num_preprocessing(test_df, num_features)

    for col in list(test_df.select_dtypes(np.number).columns):
        if col not in ('Loan_ID', 'Credit_History', 'Dependents', 'CanPay'):
            test_df[col] = test_df[col].apply(lambda x: np.log(1+x))
            test_df[f'{col}_bin'] = pd.cut(test_df[col], bins=5, labels=False)
            #test_df = test_df.drop(col, axis=1)
    #print(test_df.columns)
    
    for col in test_df.columns:
        if col not in num_features and col not in ('Loan_ID'):    
            test_df.loc[:, col] = test_df[col].astype(str).fillna('NONE')

            lbl = LabelEncoder()
            lbl.fit(test_df[col])
            test_df.loc[:, col] = lbl.transform(test_df[col])

    new_features = [
        f for f in test_df.columns if f not in ('kfold', 'Loan_Status', 'Loan_ID')
    ]

    test_preds = clf.predict(test_df[new_features])
    
    submission_df = pd.DataFrame({
            'Loan_ID': test_df['Loan_ID'],
            'Loan_Status': test_preds
        })
    
    sol_mapping = {
        0: 'N',
        1: 'Y'
    }
    submission_df.loc[:, 'Loan_Status'] = submission_df['Loan_Status'].map(sol_mapping)
    submission_df.to_csv('../models/submission_x.csv', index=False)

def run(fold, model):

    df = pd.read_csv(config.TRAINING_FILE_FOLDS)

    dependent_mapping = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3+': 3
    }
    df['Dependents'] = df['Dependents'].map(dependent_mapping)
    df['Dependents'] = df['Dependents'] + 1
    
    num_features = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Dependents'
    ]

    df[num_features] = df[num_features].fillna(df[num_features].median())
    df['Credit_History'] = df['Credit_History'].fillna(1)
    
    status_mapping = {
        'Y': 1,
        'N': 0
    }

    df.loc[:, 'Loan_Status'] = df['Loan_Status'].map(status_mapping)

    features = [
        f for f in df.columns if f not in ('kfold', 'Loan_Status', 'Loan_ID')
    ]

    cat_cols = [
        f for f in features 
            if f not in num_features
    ]

    df = feature_engineering(df, cat_cols)

    num_features = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'EMI',
        'TotalIncome',
        'IncomePerMember'
    ]

    #df = num_preprocessing(df, num_features)

    for col in list(df.select_dtypes(np.number).columns):
        if col not in ('kfold', 'Loan_Status', 'Loan_ID', 'Credit_History', 'Dependents', 'CanPay'):
            df[col] = df[col].apply(lambda x: np.log(1+x))
            df[f'{col}_bin'] = pd.cut(df[col], bins=5, labels=False)
            #df = df.drop(col, axis=1)

    for col in df.columns:
        if col not in num_features and col not in ('kfold', 'Loan_Status', 'Loan_ID'):
            df.loc[:, col] = df[col].astype(str).fillna('NONE')

            lbl = LabelEncoder()
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])
            #print(col, df[col].var())

    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_cv = df[df['kfold'] == fold].reset_index(drop=True)

    new_features = [
        f for f in df.columns if f not in ('kfold', 'Loan_Status', 'Loan_ID')
    ]

    X_train = df_train[new_features]
    X_cv = df_cv[new_features]

    y_train = df_train['Loan_Status']
    y_cv = df_cv['Loan_Status']

    #get_best_params()
    grid = {    
        'n_estimators': [200, 400, 600],
        'max_depth': [3, 7, 15],
        'criterion': ['entropy'],
        'max_features': ['sqrt', None],
        'min_samples_leaf': [2, 4],
        'min_samples_split': [5]
    }
    rf_model = RandomForestClassifier()
    model = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=grid,
        scoring='accuracy',
        n_iter=20,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    clf = model.best_estimator_

    """
    clf = model_dispatcher.models[model]
    clf.fit(X_train, y_train)
    """

    preds_cv = clf.predict(X_cv)
    accuracy = accuracy_score(y_cv, preds_cv)
    print(accuracy)
    
    """
    column_names = X_train.columns
    importances = clf.feature_importances_
    idxs = np.argsort(importances)
    for i in range(len(idxs)):
        print(column_names[i], importances[i])
    """

    test_predictions(clf, cat_cols, dependent_mapping)

if __name__ == '__main__':

    for f in range(0, 5):
        run(fold=f, model='rf')
        #run(fold=f, model='rf')        
        #run(fold=f, model='xgb')