import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges sayısal olmalı, boşluklar ve eksik değerler NaN yapılır
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Eksik değer bayrağı
    df['Missing_TotalCharges'] = df['TotalCharges'].isna().astype(int)
    df = df.dropna()
    # SeniorCitizen kategorik olarak işlenir
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')
    # Hedef değişkeni binary olarak encode et
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # customerID'yi çıkar
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    # tenure gruplama
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, np.inf], labels=['0-12', '12-24', '24-48', '48-60', '60+'])
    # toplam servis sayısı
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['TotalServices'] = df[service_cols].apply(lambda x: sum(x == 'Yes'), axis=1)
    # partner/dependents binary (zaten var ama garanti için)
    if 'Partner' in df.columns:
        df['HasPartner'] = (df['Partner'] == 'Yes').astype(int)
    if 'Dependents' in df.columns:
        df['HasDependents'] = (df['Dependents'] == 'Yes').astype(int)
    # monthly vs total charges oranı
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        df['Monthly_Total_Ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    # streaming servis toplamı
    streaming_cols = ['StreamingTV', 'StreamingMovies']
    df['TotalStreaming'] = df[streaming_cols].apply(lambda x: sum(x == 'Yes'), axis=1)
    # interaction feature: tenure * monthly charges
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['Tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    # Sözleşme tipi ve ödeme yöntemi gruplama
    if 'Contract' in df.columns:
        df['ContractType'] = df['Contract'].map({'Month-to-month': 'Short', 'One year': 'Mid', 'Two year': 'Long'})
    if 'PaymentMethod' in df.columns:
        df['PaymentAuto'] = df['PaymentMethod'].str.contains('auto', case=False, na=False).astype(int)
    return df
