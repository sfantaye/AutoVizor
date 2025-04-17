import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from scipy.stats import zscore
from pycaret.classification import setup, compare_models, pull, save_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save
from pycaret.time_series import TSForecastingExperiment
import base64
import io

st.set_page_config(page_title="AutoVizor", page_icon="🤖")
st.title("🤖 Automated EDA & AutoML Assistant")

# --- Sidebar Configs ---
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
outlier_method = st.sidebar.selectbox("Outlier Detection", ["None", "Z-score", "IQR"])
is_timeseries = st.sidebar.checkbox("Enable Time-Series Mode")

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    original_df = df.copy()

    # --- Tabs ---
    tabs = st.tabs(["🧭 Overview", "📊 EDA", "⚙️ Data Cleaning", "🧠 AutoML", "🔎 Interpretability", "⏳ Time Series"])

    # --- Tab 1: Overview ---
    with tabs[0]:
        st.subheader("🔍 Data Preview")
        st.write(df.head())
        st.write("Shape:", df.shape)
        st.write("Data Types:")
        st.write(df.dtypes)

    # --- Tab 2: EDA ---
    with tabs[1]:
        st.subheader("📊 Exploratory Data Analysis")
        st.write(df.describe())
        st.write("Missing Values:")
        st.write(df.isnull().sum())

        st.subheader("Distribution of Numeric Columns")
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # --- Tab 3: Data Cleaning ---
    with tabs[2]:
        st.subheader("🧹 Data Cleaning")

        clean_action = st.radio("Choose Cleaning Action", ["None", "Drop NA", "Impute Mean"])
        if clean_action == "Drop NA":
            df = df.dropna()
            st.success("Dropped NA values.")
        elif clean_action == "Impute Mean":
            df.fillna(df.mean(numeric_only=True), inplace=True)
            st.success("Imputed missing numeric values with mean.")

        # Outlier removal
        if outlier_method == "Z-score":
            z_scores = np.abs(zscore(df.select_dtypes(include=np.number)))
            outliers = (z_scores > 3).any(axis=1)
            df = df[~outliers]
            st.success(f"Removed {outliers.sum()} outliers using Z-score method.")

        elif outlier_method == "IQR":
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
            df = df[~outliers]
            st.success(f"Removed {outliers.sum()} outliers using IQR method.")

        # Export cleaned data
        to_download = df.to_csv(index=False)
        b64 = base64.b64encode(to_download.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">📥 Download Cleaned Data</a>'
        st.markdown(href, unsafe_allow_html=True)

    # --- Tab 4: AutoML ---
    with tabs[3]:
        st.subheader("🧠 Train Model Automatically")

        target = st.selectbox("Select Target Variable", df.columns)
        problem_type = st.radio("Problem Type", ["Classification", "Regression"])

        if st.button("Run AutoML"):
            if problem_type == "Classification":
                setup(df, target=target, silent=True, session_id=42)
                best_model = compare_models()
                st.write("Best Model:", best_model)
                model_results = pull()
                st.write(model_results)
                save_model(best_model, "best_model")

            else:
                reg_setup(df, target=target, silent=True, session_id=42)
                best_model = reg_compare()
                st.write("Best Model:", best_model)
                model_results = reg_pull()
                st.write(model_results)
                reg_save(best_model, "best_model")

    # --- Tab 5: Interpretability ---
    with tabs[4]:
        st.subheader("🔎 Model Interpretability")
        try:
            loaded_df = df.drop(columns=[target])
            loaded_model = best_model
            explainer = shap.Explainer(loaded_model.predict, loaded_df)
            shap_values = explainer(loaded_df)

            st.subheader("📈 SHAP Feature Importance")
            fig = shap.plots.bar(shap_values, show=False)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            st.error("⚠️ Train a model first to view SHAP results.")

    # --- Tab 6: Time Series ---
    with tabs[5]:
        if is_timeseries:
            st.subheader("⏳ Time Series Forecasting")
            ts_date = st.selectbox("Datetime Column", df.columns)
            ts_value = st.selectbox("Value Column", [col for col in df.columns if col != ts_date])

            df[ts_date] = pd.to_datetime(df[ts_date])
            df.set_index(ts_date, inplace=True)
            ts_df = df[[ts_value]]

            ts_exp = TSForecastingExperiment()
            ts_exp.setup(data=ts_df, session_id=123)
            best_ts_model = ts_exp.compare_models()
            future_forecast = ts_exp.predict()

            st.line_chart(future_forecast)
        else:
            st.info("Enable Time-Series Mode from the sidebar to use this feature.")


st.markdown(
    """
    <hr style="margin-top: 50px; margin-bottom: 20px;">

    <div style="text-align: center; font-size: 24px; font-weight: bold;">
        🚀 Made with ❤️ by Sintayehu Fantaye · <a href="https://github.com/sfantaye" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
