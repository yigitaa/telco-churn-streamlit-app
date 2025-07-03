import streamlit as st
import pandas as pd
import os
from src.data_download import load_telco_data
from src.data_preprocessing import clean_data, feature_engineering
from src.streamlit_inputs import get_user_inputs
from src.predict import load_pipeline_and_features, predict_single

pd.set_option('display.max_columns', None)
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    with st.expander("Dataset Column Descriptions"):
        st.markdown("""
        - **customerID**: Unique customer ID
        - **gender**: Gender of the customer
        - **SeniorCitizen**: Whether the customer is a senior citizen (1: Yes, 0: No)
        - **Partner**: Whether the customer has a partner (Yes, No)
        - **Dependents**: Whether the customer has dependents (children, parents, grandparents) (Yes, No)
        - **tenure**: Number of months the customer has stayed with the company
        - **PhoneService**: Whether the customer has phone service (Yes, No)
        - **MultipleLines**: Whether the customer has multiple lines (Yes, No, No phone service)
        - **InternetService**: Customer's internet service provider (DSL, Fiber optic, No)
        - **OnlineSecurity**: Whether the customer has online security (Yes, No, No internet service)
        - **OnlineBackup**: Whether the customer has online backup (Yes, No, No internet service)
        - **DeviceProtection**: Whether the customer has device protection (Yes, No, No internet service)
        - **TechSupport**: Whether the customer has tech support (Yes, No, No internet service)
        - **StreamingTV**: Whether the customer has streaming TV (Yes, No, No internet service) - Indicates if the customer uses internet service to stream TV from a third-party provider
        - **StreamingMovies**: Whether the customer has streaming movies (Yes, No, No internet service) - Indicates if the customer uses internet service to stream movies from a third-party provider
        - **Contract**: The contract term of the customer (Month-to-month, One year, Two year)
        - **PaperlessBilling**: Whether the customer has paperless billing (Yes, No)
        - **PaymentMethod**: Customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
        - **MonthlyCharges**: The amount charged to the customer monthly
        - **TotalCharges**: The total amount charged to the customer
        - **Churn**: Whether the customer has left the company (Yes or No) - Customers who left in the last month or quarter
        """)

st.title("Telco Customer Churn Prediction")

df = load_telco_data()
df_clean = clean_data(df)
df_feat = feature_engineering(df_clean)

# Tahmin
# st.header("Tekil Müşteri Churn Tahmini")
pipeline_path = 'models/churn_pipeline_catboost_final.pkl'
features_path = 'models/model_features.pkl'
if os.path.exists(pipeline_path) and os.path.exists(features_path):
    pipeline, model_features = load_pipeline_and_features(pipeline_path, features_path)
    input_cols = [col for col in df.columns if col not in ['customerID', 'Churn']]
    internet_service_cols = []
    phone_service_cols = []
    for col in input_cols:
        unique_vals = df[col].unique()
        if "No internet service" in unique_vals:
            internet_service_cols.append(col)
        if "No phone service" in unique_vals:
            phone_service_cols.append(col)
    main_internet_col = "InternetService" if "InternetService" in input_cols else None
    main_phone_col = "PhoneService" if "PhoneService" in input_cols else None
    special_col = "SeniorCitizen"
    st.subheader("Please enter customer informations:")
    user_input = get_user_inputs(df, input_cols, internet_service_cols, phone_service_cols, main_internet_col, main_phone_col, special_col)
    if user_input:
        btn_col1, btn_col2, btn_col3 = st.columns([2,2,6])
        with btn_col2:
            if st.button("Churn Prediction"):
                user_input[special_col] = 1 if user_input[special_col] == "Yes" else 0
                pred, proba = predict_single(user_input, pipeline, model_features)
                st.success(f"Tahmin: {'Churn' if pred == 1 else 'Not churn'}")
                st.info(f"Churn probability: {proba:.2%}")
else:
    st.warning("Final model veya feature dosyası bulunamadı. Lütfen önce modeli eğitin.")
