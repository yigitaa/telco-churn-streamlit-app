import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle config dosyasının yolu
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')

def download_telco_data():
    api = KaggleApi()
    api.authenticate()
    # Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    api.dataset_download_files('blastchar/telco-customer-churn',
                              path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
                              unzip=True)
    print('Veri indirildi ve data/ klasörüne çıkarıldı.')

def load_telco_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = pd.read_csv(data_path)
    return df

if __name__ == '__main__':
    # download_telco_data()
    df = load_telco_data()
    print(df.head())
