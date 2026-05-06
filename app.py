import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
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

# ---- Custom CSS for theme & fire animation ----
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
    .stSidebar .sidebar-content {
        background-color: rgba(34, 49, 63, 0.9);
        color: white;
    }
    h1, h2, h3 {
        color: #FF6B35;
    }
    .stMarkdown {
        color: #fff;
    }
    /* Burning number animation */
    @keyframes flicker {
        0% { text-shadow: 0 0 5px #ff8c00, 0 0 10px #ff4500, 0 0 15px #ff0000; }
        50% { text-shadow: 0 0 10px #ff8c00, 0 0 20px #ff4500, 0 0 25px #ff0000; }
        100% { text-shadow: 0 0 5px #ff8c00, 0 0 10px #ff4500, 0 0 15px #ff0000; }
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
st.markdown("""
Predict **Fire Weather Index (FWI)** using the **Algerian Forest Fires dataset**.
""")

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
    st.info("Put the Algerian dataset CSV(s) into the `dataset/` folder and reload.")
    st.stop()

st.sidebar.header("🔥 Options")
mode = st.sidebar.selectbox("Select Mode", ["EDA", "Train Model", "Predict"])

# Feature + target detection
features, _, reg_target = get_feature_target_cols(df)
st.sidebar.write(f"Detected numeric features: {len(features)}")
st.sidebar.write(features)
st.sidebar.write(f"Regression target: {reg_target}")

# ------------------------
# EDA
# ------------------------
if mode == "EDA":
    st.header("Exploratory Data Analysis 🌳")
    if st.button("Show Feature Histograms"):
        st.markdown("### 📊 Feature Distributions")
        fig, ax = plt.subplots(len(features)//3 + 1, 3, figsize=(16, 5 * (len(features)//3 + 1)))
        ax = ax.flatten()
        for i, f in enumerate(features):
            sns.histplot(df[f], ax=ax[i], kde=True, color="#FF6B35")
            ax[i].set_title(f, fontsize=12, color="#333")
        plt.tight_layout()
        st.pyplot(fig)

    if st.button("Show Correlation Heatmap"):
        st.markdown("### 🔥 Correlation Heatmap")
        num_df = df[features + ([reg_target] if reg_target else [])].copy()
        for col in num_df.columns:
            num_df[col] = pd.to_numeric(num_df[col], errors="coerce")
        num_df = num_df.dropna(axis=1, how="all")
        if num_df.shape[1] > 1:
            corr = num_df.corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠ Not enough numeric columns to compute correlation heatmap.")

# ------------------------
# Training
# ------------------------
elif mode == "Train Model":
    st.header("Train Regression Model 🔥")
    try:
        X_scaled, scaler = preprocess_for_model(df, features)
        y = pd.to_numeric(df[reg_target], errors="coerce").dropna()
        X_scaled = X_scaled.loc[y.index]
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y)

    if st.button("Train Models"):
        with st.spinner("Training regression models..."):
            reg_models = train_regression(X_train, y_train)
            st.success("✅ Models Trained: " + ", ".join(reg_models.keys()))

            for name, m in reg_models.items():
                r = evaluate_regression(m, X_test, y_test)
                st.write(f"**{name}** — RMSE: {r['rmse']:.3f}, R²: {r['r2']:.3f}")
                save_path = save_model(m, f"reg_{name}")
                st.write(f"Saved model to `{save_path}`")
                
                if hasattr(m, "feature_importances_"):
                    st.subheader(f"Feature Importance — {name}")
                    fi = pd.DataFrame({
                        "feature": features,
                        "importance": m.feature_importances_
                    }).sort_values(by="importance", ascending=False)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(data=fi, x="importance", y="feature", ax=ax, palette="YlOrRd")
                    st.pyplot(fig)

            joblib.dump(scaler, "models/scaler_reg.joblib")
            st.write("Saved scaler to `models/scaler_reg.joblib`")

# ------------------------
# Prediction
# ------------------------
elif mode == "Predict":
    st.header("Predict FWI 🔥")
    
    # Load Scaler first to ensure we have the correct feature list
    scaler = None
    if os.path.exists("models/scaler_reg.joblib"):
        scaler = joblib.load("models/scaler_reg.joblib")
        # Ensure we use the exact features the scaler was trained on
        active_features = list(scaler.feature_names_in_)
    else:
        active_features = features

    st.write("Enter feature values:")

    # Initialize input_vals with session state to allow the autofill button to work
    if "input_vals" not in st.session_state:
        st.session_state.input_vals = {f: 0.0 for f in active_features}

    # Extreme risk autofill button
    if st.button("Set Extreme Fire Risk Values"):
        for f in active_features:
            f_low = f.lower()
            if "temp" in f_low: st.session_state.input_vals[f] = 45.0
            elif "rh" in f_low: st.session_state.input_vals[f] = 10.0
            elif "ws" in f_low: st.session_state.input_vals[f] = 30.0
            elif "rain" in f_low: st.session_state.input_vals[f] = 0.0
            else: st.session_state.input_vals[f] = 80.0
        st.rerun()

    # Feature input form
    with st.form("predict_form"):
        cols = st.columns(3)
        for i, f in enumerate(active_features):
            st.session_state.input_vals[f] = cols[i % 3].number_input(f"{f}", value=st.session_state.input_vals[f], format="%.3f")
        
        submit = st.form_submit_button("Predict 🔥")

        if submit:
            X_input = pd.DataFrame([st.session_state.input_vals])

            # Scale input
            if scaler is not None:
                # FIX: Force column order and use .values to bypass name check
                X_input = X_input[active_features]
                X_scaled = scaler.transform(X_input.values)
            else:
                st.warning("Using manual scaling as scaler_reg.joblib was not found.")
                X_scaled = (X_input - X_input.mean()) / (X_input.std().replace({0:1}))
                X_scaled = X_scaled.values

            # Load regressor
            regressor = None
            for model_path in ["models/reg_random_forest_reg.joblib", "models/reg_xgboost_reg.joblib"]:
                if os.path.exists(model_path):
                    regressor = joblib.load(model_path)
                    break

            if regressor is None:
                st.error("⚠ Train a model first!")
            else:
                pred_val = float(regressor.predict(X_scaled)[0])

                # Display animated burning FWI
                st.subheader("🔥 Predicted Fire Weather Index (FWI)")
                st.markdown(f"<div class='burning'>{round(pred_val,2)}</div>", unsafe_allow_html=True)

                # Risk Logic
                if pred_val <= 5:
                    risk_level, color, desc = "Low 🔵", "blue", "Fire is unlikely."
                elif pred_val <= 12:
                    risk_level, color, desc = "Moderate 🟡", "yellow", "Stay alert."
                elif pred_val <= 20:
                    risk_level, color, desc = "High 🟠", "orange", "Take precautions."
                else:
                    risk_level, color, desc = "Extreme 🔴", "red", "Extreme fire danger!"

                st.markdown(f"**Fire Risk Level:** {risk_level}")
                st.write(desc)

                # Horizontal risk bar
                fig, ax = plt.subplots(figsize=(6, 1))
                ax.barh([0], pred_val, color=color, height=0.6)
                ax.set_xlim(0, 50)
                ax.set_yticks([])
                ax.set_xlabel("FWI scale")
                st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.write("Dataset: Algerian Forest Fires")
