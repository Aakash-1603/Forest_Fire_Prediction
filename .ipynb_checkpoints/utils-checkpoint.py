import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_dataset_from_folder(folder="dataset"):
    csv_paths = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in folder '{folder}'.")
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p, header=0)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Normalize column names
    data.columns = data.columns.str.strip().str.lower()

    # Drop unwanted columns
    for col in ["unnamed: 0", "index", "region"]:
        if col in data.columns:
            data = data.drop(columns=[col])

    return data

def basic_cleaning(df):
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(axis=1, how='all')
    df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
    return df

def get_feature_target_cols(df):
    class_target = "classes" if "classes" in df.columns else None
    reg_target = "fwi" if "fwi" in df.columns else None

    # Keep numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in [class_target, reg_target]]
    return features, class_target, reg_target

def preprocess_for_model(df, features):
    X = df[features].copy()
    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.dropna(how="any")
    if X.empty:
        raise ValueError(f"Feature DataFrame is empty after cleaning! Features requested: {features}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features, index=X.index)
    return X_scaled, scaler

def train_test_split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    if stratify is not None:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
