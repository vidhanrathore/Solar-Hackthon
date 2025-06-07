
Approach Summary

Objective:
To develop a machine learning model that predicts the performance degradation and potential failures in solar panels using sensor and maintenance data.

Tools Used:
- Python
- Pandas, NumPy for data manipulation
- CatBoost for modeling
- Scikit-learn for preprocessing and metrics

Steps Taken:

1. Data Cleaning:
   - Handled missing values optimally:
     * Mean imputation for continuous numerical values
     * Median for skewed numerical distributions
     * Mode for categorical fields

2. Feature Engineering:
   - power_output = voltage * current
   - temp_diff = module_temperature - temperature
   - irradiance_per_cloud = irradiance / (cloud_coverage + 1)

3. Categorical Encoding:
   - Used label encoding for string_id, error_code, installation_type

4. Model Training:
   - Used CatBoostRegressor with early stopping
   - Evaluation based on custom score: 100 * (1 - RMSE)

5. Output:
   - Final predictions saved in submission.csv

Note:
All steps are included in a single Python script (solar_efficiency_prediction.py)
