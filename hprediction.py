import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_excel("intern.xlsx")

# Data preprocessing and feature engineering
def preprocess_data(df):
    # Convert to datetime if not already
    df['recordedAt'] = pd.to_datetime(df['recordedAt'])
    
    # Sort by time
    df = df.sort_values('recordedAt').reset_index(drop=True)
    
    # Create time-based features
    df['hour'] = df['recordedAt'].dt.hour
    df['day_of_week'] = df['recordedAt'].dt.dayofweek
    df['month'] = df['recordedAt'].dt.month
    df['day_of_year'] = df['recordedAt'].dt.dayofyear
    
    # Create cyclical features for time elements
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Create lag features for important variables
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'temperature_lag_{lag}'] = df['temperature'].shift(lag)
        df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
    
    # Create rolling statistics
    for window in [3, 6, 12, 24]:
        df[f'temperature_rolling_mean_{window}'] = df['temperature'].rolling(window=window).mean()
        df[f'humidity_rolling_mean_{window}'] = df['humidity'].rolling(window=window).mean()
        
        df[f'temperature_rolling_std_{window}'] = df['temperature'].rolling(window=window).std()
        df[f'humidity_rolling_std_{window}'] = df['humidity'].rolling(window=window).std()
    
    # Create difference features
    df['temperature_diff'] = df['temperature'].diff()
    df['humidity_diff'] = df['humidity'].diff()
    
    # Create interaction features
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    
    # Handle missing values created by lag and rolling features
    df = df.dropna()
    
    return df

# Preprocess the data
processed_df = preprocess_data(df.copy())

# Prepare features and target
feature_columns = [
    'humidity', 'temperature', 'thi', 'light', 'heat_index', 
    'comfort', 'it', 'ratio', 'hour', 'day_of_week', 'month', 
    'day_of_year', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
]

# Add lag and rolling features
lag_features = [col for col in processed_df.columns if 'lag' in col or 'rolling' in col or 'diff' in col or 'interaction' in col]
feature_columns.extend(lag_features)

X = processed_df[feature_columns]
y = processed_df['weight']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    
    return y_pred

# 1. XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

# Plot feature importance for XGBoost
plt.figure(figsize=(10, 8))
xgb.plot_importance(xgb_model, max_num_features=15)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# 2. LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X_train, y_train)
y_pred_lgb = evaluate_model(lgb_model, X_test, y_test, "LightGBM")

# Plot feature importance for LightGBM
plt.figure(figsize=(10, 8))
lgb.plot_importance(lgb_model, max_num_features=15)
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.show()

# 3. CatBoost
cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    subsample=0.8,
    random_state=42,
    verbose=0
)

cat_model.fit(X_train, y_train)
y_pred_cat = evaluate_model(cat_model, X_test, y_test, "CatBoost")

# Plot feature importance for CatBoost
plt.figure(figsize=(10, 8))
feat_importances = pd.Series(cat_model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.show()

# Compare all models
models = {
    'XGBoost': y_pred_xgb,
    'LightGBM': y_pred_lgb,
    'CatBoost': y_pred_cat
}

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

for name, preds in models.items():
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")

# Visualization of predictions vs actual values
plt.figure(figsize=(15, 10))
plt.plot(y_test.values, label='Actual', alpha=0.7, linewidth=2)

for name, preds in models.items():
    plt.plot(preds, label=name, alpha=0.7)

plt.legend()
plt.title('Actual vs Predicted Values')
plt.xlabel('Samples')
plt.ylabel('Weight')
plt.tight_layout()
plt.show()

# Prediction function
def predict_weight(model, scaler=None):
    """
    Function to predict weight based on user input
    """
    print("\nEnter the following parameters to predict hive weight:")
    
    # Get user inputs
    temperature = float(input("🌡️  Temperature (°C): "))
    humidity = float(input("💧 Humidity (%): "))
    hour = int(input("⏱️  Hour (0-23): "))
    day_of_week = int(input("📅 Day of week (0=Mon, 6=Sun): "))
    month = int(input("📆 Month (1-12): "))
    
    # Calculate derived features
    thi = 0.8 * temperature + (temperature * humidity) / 500
    light = 100 if (hour > 6) and (hour < 18) else 0
    heat_index = temperature + 0.5 * humidity
    comfort = 70 - abs(temperature - 22) - 0.2 * abs(humidity - 50)
    it = 25.0  # Assuming constant value as in original code
    ratio = temperature / (humidity + 1)
    
    # Calculate cyclical features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Get day of year
    day_of_year = datetime.now().timetuple().tm_yday
    
    # Create input DataFrame with the same structure as training data
    input_data = {
        'humidity': [humidity],
        'temperature': [temperature],
        'thi': [thi],
        'light': [light],
        'heat_index': [heat_index],
        'comfort': [comfort],
        'it': [it],
        'ratio': [ratio],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month],
        'day_of_year': [day_of_year],
        'hour_sin': [hour_sin],
        'hour_cos': [hour_cos],
        'month_sin': [month_sin],
        'month_cos': [month_cos]
    }
    
    # Add zeros for lag and rolling features (since we don't have historical data for prediction)
    for col in X.columns:
        if col not in input_data:
            input_data[col] = [0]  
    
    # Create DataFrame with correct column order
    input_df = pd.DataFrame(input_data)[X.columns]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    print(f"🤖 PREDICTED HIVE WEIGHT: {prediction:.2f} kg")
    return prediction

predict_weight(xgb_model)