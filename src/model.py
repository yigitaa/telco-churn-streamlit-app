from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib
import os
import pandas as pd
from src.data_preprocessing import clean_data, feature_engineering

def train_model(df: pd.DataFrame, target_col: str = 'Churn'):
    X = df.drop([target_col], axis=1, errors='ignore')
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = CatBoostClassifier(verbose=0, random_state=42, iterations=300, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/churn_model.pkl')
    return model, metrics, X_test, y_test, y_pred, y_proba

def load_model(path='models/churn_model.pkl'):
    return joblib.load(path)

# Pipeline oluşturucu fonksiyon
def create_pipeline(catboost_params=None, cat_features=None):
    if catboost_params is None:
        catboost_params = {'verbose': 0, 'random_state': 42, 'iterations': 300, 'learning_rate': 0.1}
    steps = [
        ("clean_data", FunctionTransformer(clean_data)),
        ("feature_engineering", FunctionTransformer(feature_engineering)),
        ("catboost", CatBoostClassifier(**catboost_params, cat_features=cat_features))
    ]
    return Pipeline(steps)
