import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv("amr_new.csv")
df.columns = df.columns.str.strip()  

target_candidates = [col for col in df.columns if "resistance" in col.lower()]
if not target_candidates:
    raise ValueError("No column containing 'Resistance' found in dataset.")
target_column = target_candidates[0]
print(f"Target column detected: {target_column}")


def train_and_visualize_for_country(country_name):
    print(f"\n🔹 Processing data for {country_name}...")
   
    country_df = df[df["Country"] == country_name]
    if country_df.empty:
        print(f"⚠️ No data available for {country_name}")
        return
    
    X = country_df.drop(columns=[target_column])
    y = country_df[target_column]
    
    
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    
    
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols)
        ]
    )
    
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if X_train.empty or X_test.empty:
        print(f"⚠️ Not enough data to train/test for {country_name}")
        return
    
    
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    
   
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"✅ {country_name} Regression Results:")
    print("   R² Score:", r2)
    print("   MSE:", mse)
    print("   RMSE:", rmse)
    
   
    results = pd.DataFrame({
        "Actual_Resistance%": y_test.values,
        "Predicted_Resistance%": y_pred
    })
    results.to_csv(f"AMR_predictions_{country_name}.csv", index=False)
    print(f"📂 Predictions saved to 'AMR_predictions_{country_name}.csv'")
    
    
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolor='k')
    plt.plot([0, 100], [0, 100], color='red', linestyle='--', linewidth=2, label="Perfect Prediction")
    plt.xlabel("Actual Resistance%")
    plt.ylabel("Predicted Resistance%")
    plt.title(f"Actual vs Predicted Resistance% ({country_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

for country in ["India", "Usa", "Uk"]:
    train_and_visualize_for_country(country)
