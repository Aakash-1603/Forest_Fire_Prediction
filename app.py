import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Import custom modules from your repo
from utils import (
    load_dataset_from_folder,
    basic_cleaning,
    get_feature_target_cols,
    preprocess_for_model,
    train_test_split_data,
)
from models import (
    train_regression,
    evaluate_regression,
    save_model,
)

st.set_page_config(
    page_title="🌲 Forest Fire Predictor",
    layout="wide",
    page_icon="🔥"
)

# ---- Custom CSS for styling ----
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://unsplash.com");
        background-size: cover;
        background-attachment: fixed;
    }
    .stButton>button {
        background-color: #FF6B35;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    h1, h2, h3 { color: #FF6B35; }
    .stMarkdown { color: #fff; }
    @keyframes flicker {
        0% { text-shadow: 0 0 5px #ff8c00, 0 0 10px #ff4500; }
        100% { text-shadow: 0 0 10px #ff8c00, 0 0 25px #ff0000; }
    }
    .burning {
        animation: flicker 1s infinite alternate;
        font-size: 48px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌲 Forest Fire Prediction")

# ------------------------
# Load dataset
# ------------------------
@st.cache_data
def load_data():
    df = load_dataset_from_folder("dataset")
    df = basic_cleaning(df)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Dataset load error: {e}")
    st.stop()

st.sidebar.header("🔥 Options")
mode = st.sidebar.selectbox("Select Mode", ["EDA", "Train Model", "Predict"])

# Detection for Training/EDA
features, _, reg_target = get_feature_target_cols(df)

# ------------------------
# EDA Mode
# ------------------------
if mode == "EDA":
    st.header("Exploratory Data Analysis 🌳")
    if st.button("Show Feature Histograms"):
        fig, ax = plt.subplots(len(features)//3 + 1, 3, figsize=(16, 5 * (len(features)//3 + 1)))
        ax = ax.flatten()
        for i, f in enumerate(features):
            sns.histplot(df[f], ax=ax[i], kde=True, color="#FF6B35")
        plt.tight_layout()
        st.pyplot(fig)

# ------------------------
# Training Mode
# ------------------------
elif mode == "Train Model":
    st.header("Train Regression Model 🔥")
    try:
        X_scaled, scaler = preprocess_for_model(df, features)
        y = pd.to_numeric(df[reg_target], errors="coerce").dropna()
        X_scaled = X_scaled.loc[y.index]
        X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y)

        if st.button("Start Training"):
            with st.spinner("Training..."):
                reg_models = train_regression(X_train, y_train)
                for name, m in reg_models.items():
                    r = evaluate_regression(m, X_test, y_test)
                    st.write(f"**{name}** — RMSE: {r['rmse']:.3f}, R²: {r['r2']:.3f}")
                    save_model(m, f"reg_{name}")
                
                joblib.dump(scaler, "models/scaler_reg.joblib")
                st.success("✅ Training Complete. Scaler & Model saved in /models folder.")
    except Exception as e:
        st.error(f"Training error: {e}")

# ------------------------
# Prediction Mode (FIXED)
# ------------------------
elif mode == "Predict":
    st.header("Predict FWI 🔥")
    
    # Ensure Scaler exists to determine input features
    if os.path.exists("models/scaler_reg.joblib"):
        scaler = joblib.load("models/scaler_reg.joblib")
        # Critical Fix: Always use features the scaler was actually trained on
        prediction_features = list(scaler.feature_names_in_)
    else:
        st.error("⚠ No scaler found. Go to 'Train Model' mode first.")
        st.stop()

    # Form for Inputs
    input_vals = {}
    with st.form("predict_form"):
        st.write("Enter Weather Metrics:")
        cols = st.columns(3)
        for i, f in enumerate(prediction_features):
            input_vals[f] = cols[i % 3].number_input(f"{f}", value=0.0, format="%.2f")
        
        submit = st.form_submit_button("Predict 🔥")

        if submit:
            # 1. Create DF with EXACT order and names
            X_input = pd.DataFrame([input_vals])[prediction_features]
            
            # 2. Transform using .values to bypass sklearn's feature name validation
            X_scaled = scaler.transform(X_input.values)

            # 3. Load the best available regressor
            regressor = None
            for m_file in ["reg_random_forest_reg.joblib", "reg_xgboost_reg.joblib", "reg_linear_regression.joblib"]:
                path = f"models/{m_file}"
                if os.path.exists(path):
                    regressor = joblib.load(path)
                    break

            if regressor:
                pred_val = float(regressor.predict(X_scaled)[0])
                
                # Result Visualization
                st.subheader("🔥 Predicted Fire Weather Index (FWI)")
                st.markdown(f"<div class='burning'>{round(pred_val,2)}</div>", unsafe_allow_html=True)
                
                # Risk Logic
                risk = "Low 🔵" if pred_val <= 5 else "Moderate 🟡" if pred_val <= 12 else "High 🟠" if pred_val <= 20 else "Extreme 🔴"
                st.info(f"**Fire Risk Level:** {risk}")
            else:
                st.error("⚠ No model file found. Please train a model first.")

st.sidebar.markdown("---")
st.sidebar.write("Developed for Forest Fire Analysis")
