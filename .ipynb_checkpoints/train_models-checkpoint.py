import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("dataset/Algerian_forest_fires_dataset_UPDATE.csv", header=1)

# Drop rows that are just region names (non-numeric)
df = df[~df["day"].astype(str).str.contains("Region", na=False)]

# Drop duplicate headers (like 'day' repeating inside file)
df = df[df["day"] != "day"]

# Normalize column names (strip spaces, lowercase)
df.columns = df.columns.str.strip().str.lower()

# Reset index
df = df.reset_index(drop=True)

# -------------------------------
# 2. Clean numeric columns
# -------------------------------
for col in df.columns:
    # Remove spaces inside numbers (e.g., "14.6 9" -> "14.69")
    df[col] = df[col].astype(str).str.replace(" ", "", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop completely empty columns if any
df = df.dropna(axis=1, how="all")

# Drop rows with missing target (FWI)
df = df.dropna(subset=["fwi"])

# -------------------------------
# 3. Features and target
# -------------------------------
X = df.drop(["fwi"], axis=1)
y = df["fwi"]

# -------------------------------
# 4. Scale features
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 5. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Train regression model
# -------------------------------
regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
regressor.fit(X_train, y_train)

# -------------------------------
# 7. Save model + scaler
# -------------------------------
joblib.dump(regressor, "models/regressor.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Regression model trained and saved in 'models/' folder")
