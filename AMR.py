import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
# ------------------------------
# 1️⃣ Load Dataset
# ------------------------------
df = pd.read_csv("amr_new.csv")
df.columns = df.columns.str.strip()  # Clean column names

# ------------------------------
# 2️⃣ Detect target column
# ------------------------------
target_candidates = [col for col in df.columns if "resistance" in col.lower()]
if not target_candidates:
    raise ValueError("No column containing 'Resistance' found in dataset.")
target_column = target_candidates[0]
print(f"Target column detected: {target_column}")

# ------------------------------
# 🔹 Function to Train & Visualize by State (within a country)
# ------------------------------
def train_and_visualize_by_state(country_name):
    print(f"\n🌍 Processing data for {country_name}...")
    
    # Filter dataset for specific country
    country_df = df[df["Country"].str.lower() == country_name.lower()]
    if country_df.empty:
        print(f"⚠️ No data available for {country_name}")
        return
    
    # Get all states
    states = country_df["State"].dropna().unique()
    
    for state in states:
        print(f"\n🔹 Training model for {country_name} - {state}...")
        state_df = country_df[country_df["State"] == state]
        
        if len(state_df) < 10:  # too little data for training
            print(f"⚠️ Skipping {state} (not enough samples: {len(state_df)})")
            continue
        
        # Features & target
        X = state_df.drop(columns=[target_column])
        y = state_df[target_column]
        
        # Separate categorical & numeric
        categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
        numeric_cols = [col for col in X.columns if col not in categorical_cols]
        
        # Preprocessing
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        numeric_transformer = StandardScaler()
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, categorical_cols),
                ("num", numeric_transformer, numeric_cols)
            ]
        )
        
        # Pipeline with RandomForest
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if X_train.empty or X_test.empty:
            print(f"⚠️ Not enough data to train/test for {state}")
            continue
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"✅ {country_name}-{state} Results:")
        print("   R² Score:", r2)
        print("   MSE:", mse)
        print("   RMSE:", rmse)
        
        # Save results to CSV
        results = pd.DataFrame({
            "Actual_Resistance%": y_test.values,
            "Predicted_Resistance%": y_pred
        })
        results.to_csv(f"AMR_predictions_{country_name}_{state}.csv", index=False)
        
        # Visualization
        plt.figure(figsize=(7,5))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolor='k')
        plt.plot([0, 100], [0, 100], color='red', linestyle='--', linewidth=2, label="Perfect Prediction")
        plt.xlabel("Actual Resistance%")
        plt.ylabel("Predicted Resistance%")
        plt.title(f"Actual vs Predicted Resistance% ({country_name} - {state})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

# ------------------------------
# 🔹 Example: Train for India
# ------------------------------
train_and_visualize_by_state("India")
train_and_visualize_by_state("Uk")
train_and_visualize_by_state("Usa")

