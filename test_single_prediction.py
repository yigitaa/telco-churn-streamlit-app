import pandas as pd
from src.data_download import load_telco_data
from src.data_preprocessing import clean_data, feature_engineering
from src.predict import load_pipeline_and_features, predict_single

# 1. Veriyi yükle
pipeline_path = 'models/churn_pipeline_catboost_final.pkl'
features_path = 'models/model_features.pkl'

df = load_telco_data()
df.info()
# 2. İlgili müşteri satırını seç
row = df[df['customerID'] == '8919-FYFQZ']

# 3. Temizle ve feature engineering uygula
row_clean = clean_data(row)
row_feat = feature_engineering(row_clean)

# 4. Model ve feature'ları yükle
pipeline, model_features = load_pipeline_and_features(pipeline_path, features_path)

# 5. Sadece modelde kullanılan feature'ları al
X = row_feat[model_features[:]]  # Son eleman 'Churn' ise onu çıkar
X.info()
# 6. Tahmin yap
pred, proba = predict_single(X.iloc[0].to_dict(), pipeline, model_features)

# 7. Sonucu yazdır
print(f"Prediction: {pred}")
print(f"Probability: {proba:.4f}")

bos_total_charges = df[df['TotalCharges'].isna()]
print(bos_total_charges)

import numpy as np

df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = df['TotalCharges'].astype(float)