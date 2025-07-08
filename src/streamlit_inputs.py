# streamlit_inputs.py
# Her sütun için arayüzde gösterilecek selectbox seçeneklerini burada tanımlayabilirsiniz.
# Gerekirse diğer sütunlar için de ekleme yapabilirsiniz.

import streamlit as st

def get_all_selectbox_columns(df):
    """
    DataFrame'de dtype'ı object veya kategorik olan tüm sütunları ve unique seçeneklerini döndürür.
    """
    selectbox_dict = {}
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            selectbox_dict[col] = df[col].unique()
    return selectbox_dict

def get_user_inputs(df, input_cols, internet_service_cols, phone_service_cols, main_internet_col, main_phone_col, special_col, labels=None):
    """
    Streamlit arayüzünde kullanıcıdan inputları alır ve bir dict olarak döndürür.
    TotalCharges input olarak alınmaz, otomatik hesaplanır ve sadece gösterilir.
    labels: input_cols ile aynı sırada, gösterilecek label isimleri (opsiyonel)
    """
    user_input = {}
    cols = st.columns(2)
    if main_internet_col:
        label = labels[input_cols.index(main_internet_col)] if labels else main_internet_col
        user_input[main_internet_col] = cols[0].selectbox(label, options=df[main_internet_col].unique(), key=main_internet_col)
    if main_phone_col:
        label = labels[input_cols.index(main_phone_col)] if labels else main_phone_col
        user_input[main_phone_col] = cols[1].selectbox(label, options=df[main_phone_col].unique(), key=main_phone_col)
    if user_input.get(main_internet_col) is not None and user_input.get(main_phone_col) is not None:
        cols = st.columns(3)
        label = labels[input_cols.index(special_col)] if labels else special_col
        user_input[special_col] = cols[0].selectbox(label, options=["No", "Yes"], key=special_col)
        col_idx = 1
        for i, col in enumerate(input_cols):
            if col in [main_internet_col, main_phone_col, special_col, "TotalCharges"]:
                continue
            current_col = cols[col_idx % 3]
            disabled = False
            if col in internet_service_cols and user_input.get(main_internet_col, None) == "No":
                default_val = "No internet service"
                disabled = True
            elif col in phone_service_cols and user_input.get(main_phone_col, None) == "No":
                default_val = "No phone service"
                disabled = True
            else:
                default_val = df[col].unique()[0]
            label = labels[i] if labels else col
            if col == "tenure":
                user_input[col] = int(current_col.number_input(label, value=int(df[col].mean()), key=col, disabled=disabled, step=1, format="%d", min_value=1))
            elif col == "MonthlyCharges":
                user_input[col] = float(current_col.number_input(label, value=float(df[col].mean()), key=col, disabled=disabled, format="%.2f", min_value=1.0))
            elif df[col].dtype == 'object' or str(df[col].dtype) == 'category':
                user_input[col] = current_col.selectbox(label, options=df[col].unique(), key=col, index=list(df[col].unique()).index(default_val), disabled=disabled)
            else:
                user_input[col] = current_col.number_input(label, value=float(df[col].mean()), key=col, disabled=disabled)
            col_idx += 1
        # TotalCharges'ı inputlardan sonra ayrı olarak göster
        tenure_val = user_input.get("tenure", int(df["tenure"].mean()) if "tenure" in df.columns else 0)
        monthly_val = user_input.get("MonthlyCharges", float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df.columns else 0)
        total_val = float(tenure_val) * float(monthly_val)
        user_input["TotalCharges"] = total_val
        st.markdown("<b>TotalCharges</b>", unsafe_allow_html=True)
        st.number_input("TotalCharges", value=total_val, key="totalcharges_display", disabled=True, format="%.2f")
        st.markdown("<br>", unsafe_allow_html=True)
    return user_input
