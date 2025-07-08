import pandas as pd
import joblib
from src.data_preprocessing import clean_data, feature_engineering

def load_pipeline_and_features(pipeline_path, features_path):
    pipeline = joblib.load(pipeline_path)
    model_features = joblib.load(features_path)
    return pipeline, model_features

def predict_single(input_dict, pipeline, model_features):
    df = pd.DataFrame([input_dict])
    df_clean = clean_data(df)
    df_feat = feature_engineering(df_clean)
    for feat in model_features:
        if feat not in df_feat.columns:
            df_feat[feat] = 0
    df_feat = df_feat[model_features]
    cat_features = df_feat.select_dtypes(include=['object', 'category']).columns
    for col in cat_features:
        df_feat[col] = df_feat[col].astype(str)
    pred = pipeline.predict(df_feat)[0]
    proba = pipeline.predict_proba(df_feat)[0,1]
    return pred, proba

def get_shap_values(input_dict, pipeline, model_features):
    from catboost import Pool
    import numpy as np
    df = pd.DataFrame([input_dict])
    df_clean = clean_data(df)
    df_feat = feature_engineering(df_clean)
    for feat in model_features:
        if feat not in df_feat.columns:
            df_feat[feat] = 0
    df_feat = df_feat[model_features]
    cat_features = df_feat.select_dtypes(include=['object', 'category']).columns
    for col in cat_features:
        df_feat[col] = df_feat[col].astype(str)
    model = pipeline.named_steps['catboost'] if 'catboost' in pipeline.named_steps else pipeline
    pool = Pool(df_feat, cat_features=list(cat_features))
    shap_values = model.get_feature_importance(data=pool, type='ShapValues')
    shap_vals = shap_values[0][:-1]
    base_value = shap_values[0][-1]
    feature_names = df_feat.columns.tolist()
    return shap_vals, feature_names, base_value
