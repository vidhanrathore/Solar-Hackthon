
# solar_efficiency_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Data Cleaning
train.fillna({
    'temperature': train['temperature'].mean(),
    'irradiance': train['irradiance'].mean(),
    'panel_age': train['panel_age'].median(),
    'maintenance_count': train['maintenance_count'].median(),
    'soiling_ratio': train['soiling_ratio'].median(),
    'voltage': train['voltage'].mean(),
    'current': train['current'].mean(),
    'module_temperature': train['module_temperature'].mean(),
    'cloud_coverage': train['cloud_coverage'].median(),
    'error_code': train['error_code'].mode()[0],
    'installation_type': train['installation_type'].mode()[0]
}, inplace=True)

test.fillna({
    'temperature': test['temperature'].mean(),
    'irradiance': test['irradiance'].mean(),
    'panel_age': test['panel_age'].median(),
    'maintenance_count': test['maintenance_count'].median(),
    'soiling_ratio': test['soiling_ratio'].median(),
    'voltage': test['voltage'].mean(),
    'current': test['current'].mean(),
    'module_temperature': test['module_temperature'].mean(),
    'cloud_coverage': test['cloud_coverage'].median(),
    'error_code': test['error_code'].mode()[0],
    'installation_type': test['installation_type'].mode()[0]
}, inplace=True)

# Feature Engineering
for df in [train, test]:
    df['power_output'] = df['voltage'] * df['current']
    df['temp_diff'] = df['module_temperature'] - df['temperature']
    df['irradiance_per_cloud'] = df['irradiance'] / (df['cloud_coverage'] + 1)

# Encode categorical features
cat_cols = ['string_id', 'error_code', 'installation_type']
for col in cat_cols:
    train[col] = train[col].astype('category').cat.codes
    test[col] = test[col].astype('category').cat.codes

# Feature selection
drop_cols = ['id', 'efficiency']
X = train.drop(columns=drop_cols)
y = train['efficiency']
X_test = test.drop(columns=['id'])

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = CatBoostRegressor(iterations=150, learning_rate=0.1, depth=6, random_seed=42, verbose=False)
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

# Validation score
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
score = (1 - rmse) * 100
print(f"RMSE: {rmse:.4f}, Score: {score:.2f}")

# Final prediction
preds = model.predict(X_test)
submission = pd.DataFrame({'id': test['id'], 'efficiency': preds})
submission.to_csv('submission.csv', index=False)
