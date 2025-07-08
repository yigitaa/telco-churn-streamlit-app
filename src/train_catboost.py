import os
from src.data_download import load_telco_data
from src.data_preprocessing import clean_data, feature_engineering
from src.model import create_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
import joblib
import optuna
import numpy as np
from tqdm import tqdm
from sklearn.base import clone

df = load_telco_data()
df_clean = clean_data(df)
df_feat = feature_engineering(df_clean)

df.info()
df[df['customerID']=='9505-SQFSW']
# cat_features ve X, feature engineering sonrası df_feat üzerinden oluşturulmalı
cat_features = [i for i, col in enumerate(df_feat.drop(['Churn'], axis=1, errors='ignore').columns)
                if (df_feat[col].dtype == 'object' or str(df_feat[col].dtype)=='category')]
print(f"CatBoost cat_features indexleri: {cat_features}")

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1500, step=100),
        'loss_function': 'Logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 6),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.2, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'random_state': 42,
        'verbose': 0
    }
    pipeline = create_pipeline(catboost_params=params, cat_features=cat_features)
    scores = []
    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        pipeline_clone = clone(pipeline)
        pipeline_clone.fit(X_train, y_train)
        y_proba = pipeline_clone.predict_proba(X_valid)[:,1]
        score = roc_auc_score(y_valid, y_proba)
        scores.append(score)
    return np.mean(scores)
df_feat.columns
X = df_feat.drop(['Churn'], axis=1, errors='ignore')
y = df_feat['Churn']
joblib.dump(X.columns.tolist(), 'models/model_features.pkl')

# StratifiedKFold ile cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

X.info()
n_trials = 20
with tqdm(total=n_trials, desc="Optuna Trials") as pbar:
    def tqdm_callback(study, trial):
        pbar.update(1)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])
print(f"En iyi parametreler: {study.best_params}")
print(f"En iyi ROC AUC skoru: {study.best_value:.4f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
best_pipeline = create_pipeline(catboost_params=study.best_params, cat_features=cat_features)
best_pipeline.fit(X_train, y_train)

y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:,1]
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# Pipeline'ı kaydet
os.makedirs('models', exist_ok=True)
joblib.dump(best_pipeline, 'models/churn_pipeline_catboost.pkl')
print("Pipeline 'models/churn_pipeline_catboost.pkl' olarak kaydedildi.")

final_pipeline = create_pipeline(catboost_params=study.best_params, cat_features=cat_features)
final_pipeline.fit(X, y)
joblib.dump(final_pipeline, 'models/churn_pipeline_catboost_final.pkl')
print("Tüm veriyle eğitilmiş final model 'models/churn_pipeline_catboost_final.pkl' olarak kaydedildi.")


