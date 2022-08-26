import config
import model_dispatcher
import itertools
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

def feature_engineering(df, cat_cols):

    combs = list(itertools.combinations(cat_cols, 2))

    for c1, c2 in combs:
        df.loc[:, c1+'_'+c2] = df[c1].astype(str) + '_' + df[c2].astype(str)

    return df

def run(fold, model):
    df = pd.read_csv(config.TRAINING_FILE_FOLDS)

    num_features = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term'
    ]

    status_mapping = {
        'Y': 1,
        'N': 0
    }

    df.loc[:, 'Loan_Status'] = df['Loan_Status'].map(status_mapping)

    # fill integer missing values
    df.loc[:, num_features] = df[num_features].fillna(df[num_features].median())

    features = [
        f for f in df.columns if f not in ('kfold', 'Loan_Status', 'Loan_ID')
    ]

    cat_cols = [
        f for f in df.columns 
            if f not in ('kfold', 'Loan_Status', 'Loan_ID') and
            f not in num_features
    ]

    for col in cat_cols:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')

    df = feature_engineering(df, cat_cols)

    cat_cols = [
        f for f in df.columns 
            if f not in ('kfold', 'Loan_Status', 'Loan_ID') and
            f not in num_features
    ]

    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_cv = df[df['kfold'] == fold].reset_index(drop=True)
    
    X_train = df_train[num_features]
    X_cv = df_cv[num_features]

    full_cat_data = pd.concat(
        [df_train[cat_cols], df_cv[cat_cols]],
        axis=0
    ) 

    ohe = OneHotEncoder()
    ohe.fit(full_cat_data)

    cat_train = pd.DataFrame(ohe.transform(df_train[cat_cols]).toarray())
    cat_cv = pd.DataFrame(ohe.transform(df_cv[cat_cols]).toarray())

    X_train = X_train.join(cat_train)
    X_cv = X_cv.join(cat_cv)

    y_train = df_train['Loan_Status']
    y_cv = df_cv['Loan_Status']

    clf = model_dispatcher.models[model]
    clf.fit(X_train, y_train)

    preds_cv = clf.predict(X_cv)
    accuracy = accuracy_score(y_cv, preds_cv)
    print(accuracy)

    test_df = pd.read_csv(config.TEST_FILE)
    test_df[num_features] = test_df[num_features].fillna(test_df[num_features].median())

    cat_cols = [
        f for f in test_df.columns 
            if f not in ('Loan_ID') and
            f not in num_features
    ]

    for col in cat_cols:
        test_df.loc[:, col] = test_df[col].astype(str).fillna('NONE')

    test_df = feature_engineering(test_df, cat_cols)

    print(test_df.columns)

    cat_cols = [
        f for f in test_df.columns 
            if f not in ('Loan_ID') and
            f not in num_features
    ]

    cat_test = pd.DataFrame(ohe.transform(test_df[cat_cols]).toarray())
    
    X_test = test_df[num_features].join(cat_test)

    test_preds = clf.predict(X_test)
    
    submission_df = pd.DataFrame({
            'Loan_ID': test_df['Loan_ID'],
            'Loan_Status': test_preds
        })
    
    sol_mapping = {
        0: 'N',
        1: 'Y'
    }
    submission_df.loc[:, 'Loan_Status'] = submission_df['Loan_Status'].map(sol_mapping)
    submission_df.to_csv('../models/submission_x2.csv', index=False)

if __name__ == '__main__':

    for f in range(0, 5):
        run(fold=f, model='rf')