from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

models = {
    'lr': LogisticRegression(),
    'rf': RandomForestClassifier(),
    'xgb': xgb.XGBClassifier(
        eval_metric='error'
        )
}