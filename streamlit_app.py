import streamlit as st
import pandas as pd
import os
from src.data_download import load_telco_data
from src.data_preprocessing import clean_data, feature_engineering
from src.streamlit_inputs import get_user_inputs
from src.predict import load_pipeline_and_features, predict_single, get_shap_values

pd.set_option('display.max_columns', None)
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.title("Telco Customer Churn Prediction")

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    with st.expander(":blue[**Dataset Column Descriptions**]", expanded=True):
        st.markdown("""
        <style>
        .desc-col ul {margin-bottom: 0;}
        </style>
        """, unsafe_allow_html=True)
        desc1 = [
            "**customerID**: Unique customer ID",
            "**gender**: Gender of the customer",
            "**SeniorCitizen**: Whether the customer is a senior citizen (1: Yes, 0: No)",
            "**Partner**: Whether the customer has a partner (Yes, No)",
            "**Dependents**: Whether the customer has dependents (Yes, No)",
            "**tenure**: Number of months the customer has stayed with the company",
            "**PhoneService**: Whether the customer has phone service (Yes, No)",
            "**MultipleLines**: Whether the customer has multiple lines (Yes, No, No phone service)",
            "**InternetService**: Customer's internet service provider (DSL, Fiber optic, No)",
            "**OnlineSecurity**: Whether the customer has online security (Yes, No, No internet service)",
        ]
        desc2 = [
            "**OnlineBackup**: Whether the customer has online backup (Yes, No, No internet service)",
            "**DeviceProtection**: Whether the customer has device protection (Yes, No, No internet service)",
            "**TechSupport**: Whether the customer has tech support (Yes, No, No internet service)",
            "**StreamingTV**: Whether the customer has streaming TV (Yes, No, No internet service)",
            "**StreamingMovies**: Whether the customer has streaming movies (Yes, No, No internet service)",
            "**Contract**: The contract term of the customer (Month-to-month, One year, Two year)",
            "**PaperlessBilling**: Whether the customer has paperless billing (Yes, No)",
            "**PaymentMethod**: Customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))",
            "**MonthlyCharges**: The amount charged to the customer monthly",
            "**TotalCharges**: The total amount charged to the customer",
            "**Churn**: Whether the customer has left the company (Yes or No)",
        ]
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown("\n".join([f"- {item}" for item in desc1]), unsafe_allow_html=True)
        with dcol2:
            st.markdown("\n".join([f"- {item}" for item in desc2]), unsafe_allow_html=True)

df = load_telco_data()
df_clean = clean_data(df)
df_feat = feature_engineering(df_clean)

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
    # Yanyana iki sütun: sol input, sağ prediction analysis
    input_col, analysis_col = st.columns(2)
    # --- Feature name formatting function ---
    import re
    def format_label(col_name):
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', col_name)
        return ' '.join([w.capitalize() for w in words])
    formatted_input_cols = [format_label(col) for col in input_cols]
    with input_col:
        st.subheader("Please enter customer informations:")
        user_input = get_user_inputs(df, input_cols, internet_service_cols, phone_service_cols, main_internet_col, main_phone_col, special_col, labels=formatted_input_cols)
        predict_clicked = st.button("Churn Prediction")
    if user_input and predict_clicked:
        user_input[special_col] = 1 if user_input[special_col] == "Yes" else 0
        pred, proba = predict_single(user_input, pipeline, model_features)
        with analysis_col:
            st.subheader("Prediction Analysis")
            if pred == 1:
                st.error(f"Prediction: Churn", icon="⚠️")
            else:
                st.success(f"Prediction: Not churn", icon="✅")
            st.info(f"Churn probability: {proba:.2%}")
            # SHAP değerlerini hesapla ve göster
            shap_vals, feature_names, base_value = get_shap_values(user_input, pipeline, model_features)
            import pandas as pd
            import numpy as np
            shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_vals})
            shap_df['abs_val'] = np.abs(shap_df['SHAP Value'])
            shap_df = shap_df.sort_values('abs_val', ascending=False).head(10)
            st.markdown("**Top 10 Features Affecting Churn Probability:**")
            # Bar chart (horizontal) - çok büyük boyut
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(24, 16))
            shap_df = shap_df.sort_values('SHAP Value', ascending=True)
            ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color='skyblue')
            ax.set_xlabel('SHAP Value (Effect)', fontsize=22)
            ax.set_ylabel('Feature', fontsize=22)
            ax.set_title('Top 10 Features Affecting Churn Probability', fontsize=26)
            ax.tick_params(axis='both', labelsize=20)
            st.pyplot(fig)
            # --- Similar Customers Analysis ---
            st.subheader("Similar Customers")
            st.markdown("The customers most similar to the user input are listed below.")
            # Kategorik ve sayısal sütunları ayır
            cat_cols = df_feat.select_dtypes(include=['object', 'category']).columns.tolist()
            num_cols = df_feat.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
            # Kullanıcı girdisini ve veri setini aynı şekilde encode et
            from sklearn.preprocessing import OneHotEncoder
            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            except TypeError:
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            # Sadece modelde kullanılan feature'ları al
            features_for_sim = [col for col in model_features if col != 'Churn']
            df_sim = df_feat[features_for_sim].copy()
            # Kullanıcı girdisini DataFrame'e çevir ve aynı pipeline'dan geçir
            user_input_df = pd.DataFrame([user_input])
            user_input_clean = clean_data(user_input_df)
            user_input_feat = feature_engineering(user_input_clean)
            # Sadece modelde kullanılan feature'ları al
            user_df = user_input_feat[features_for_sim]
            # Birleştirip encode et
            df_all = pd.concat([user_df, df_sim], ignore_index=True)
            df_encoded = encoder.fit_transform(df_all)
            # İlk satır kullanıcı, diğerleri dataset
            user_vec = df_encoded[0]
            data_vecs = df_encoded[1:]
            from sklearn.metrics.pairwise import euclidean_distances
            dists = euclidean_distances([user_vec], data_vecs)[0]
            top_n = 5
            top_idx = dists.argsort()[:top_n]
            similar_customers = df_sim.iloc[top_idx].copy()
            similar_customers['Distance'] = dists[top_idx]
            # Orijinal df ile customerID ve Churn bilgisini ekle
            similar_customers = similar_customers.merge(df[['customerID', 'Churn']], left_index=True, right_index=True)
            # Show user input summary in the same format as similar customers
            user_churn = 'Yes' if pred == 1 else 'No'
            st.markdown("<div style='margin-bottom: 12px; background-color:#222; color:#fff; padding:8px; border-radius:6px;'>"
                        f"<b>User input</b> &nbsp;|&nbsp; "
                        f"<b>Churn:</b> {user_churn} &nbsp;|&nbsp; "
                        f"<b>Tenure:</b> {user_input.get('tenure', '-')} months &nbsp;|&nbsp; "
                        f"<b>Contract:</b> {user_input.get('Contract', '-')} &nbsp;|&nbsp; "
                        f"<b>Monthly Charge:</b> ${float(user_input.get('MonthlyCharges', 0)):.2f} &nbsp;|&nbsp; "
                        f"<b>Total Charge:</b> ${float(user_input.get('TotalCharges', 0)):.2f}"
                        "</div>", unsafe_allow_html=True)
            # Display summary for each similar customer
            for idx, row in similar_customers.iterrows():
                st.markdown(f"<div style='margin-bottom: 12px;'>"
                            f"<b>CustomerID:</b> {row['customerID']} &nbsp;|&nbsp; "
                            f"<b>Churn:</b> {row['Churn']} &nbsp;|&nbsp; "
                            f"<b>Tenure:</b> {row['tenure']} months &nbsp;|&nbsp; "
                            f"<b>Contract:</b> {row['Contract']} &nbsp;|&nbsp; "
                            f"<b>Monthly Charge:</b> ${float(row['MonthlyCharges']):.2f} &nbsp;|&nbsp; "
                            f"<b>Total Charge:</b> ${float(row['TotalCharges']):.2f}"
                            f"</div>", unsafe_allow_html=True)
else:
    st.warning("Final model veya feature dosyası bulunamadı. Lütfen önce modeli eğitin.")
