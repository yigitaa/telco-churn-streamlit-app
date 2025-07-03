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
