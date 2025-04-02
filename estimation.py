"""
estimation.py - Estimation of Game-Playing Time from Steam Reviews

This script performs descriptive data mining on video game reviews from the Steam platform.
- Loads and preprocesses a large dataset of game reviews.
- Engineers features such as review word count and early access flag.
- Splits the data into training and development subsets.
- Implements various regression models including:
    - Baseline Linear Regression
    - Linear Regression with Outlier Removal
    - Log-Transformed Target Regression
    - ElasticNet with Hyperparameter Tuning
- Saves and reloads the best-performing pipeline for further testing.

Acknowledgements:
This project is part of a Computer Science course on Data Mining.
Special thanks to the CSC373 faculty for guidance and for providing the dataset.

Author: Dario Santiago Lopez, Anthony Roca, and ChatGPT
Date: April 2, 2025
"""

import os
import json
import gzip
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, ElasticNet, Lasso 
from sklearn.metrics import mean_squared_error 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import pickle 
import time 

# -------------------------------
# 1. Data Loading
# -------------------------------
print("Estimation.py")
print("--------------------")
print("Loading data...")

# Start timer for data loading 
data_start = time.time()

# code to open gzipped json file and read reviews into a list
input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")
dataset = []
for l in input_file:
    d = eval(l)
    dataset.append(d)
input_file.close()

# Measure and print data loading time
data_end = time.time() - data_start
print(f"Data loaded in {data_end:.2f} seconds.\n")

# Convert the list to a DataFrame 
df = pd.DataFrame(dataset) 

# -------------------------------
# 2. Data Split
# -------------------------------
# code to split the data (decided to use Pandas since it's more convenient for feature engineering)
split_idx = int(len(df) * 0.8) 
train_data = df.iloc[:split_idx].copy() 
dev_data = df.iloc[split_idx:].copy() 

# Extract year from the date (decided to use Pandas for better performance and error handling)
train_data['year'] = pd.to_datetime(train_data['date']).dt.year 
dev_data['year'] = pd.to_datetime(dev_data['date']).dt.year 

# -------------------------------
# 3. Feature Engineering 
# -------------------------------
# Compute review word count as a feature 
train_data['review_word_count'] = train_data['text'].apply(lambda x: len(x.split()))
dev_data['review_word_count'] = dev_data['text'].apply(lambda x: len(x.split()))

# Convert early_access to binary feature (boolean)
train_data['early_access'] = train_data['early_access'].astype(int) 
dev_data['early_access'] = dev_data['early_access'].astype(int) 

# Feature selection we may want to use 
features = ['review_word_count', 'products', 'page_order', 'early_access', 'page']   # Removed 'username' and 'product_id'
target = 'hours' 

# Drop rows with missing values in features or target 
train_data = train_data.dropna(subset = [target]) # Originally had (subset = features + [target])
dev_data = dev_data.dropna(subset = [target])

# Set up imputer for missing values 
imputer = SimpleImputer(strategy = 'mean')

# Fit imputer on training data 
X_train = imputer.fit_transform(train_data[features].values)
X_dev = imputer.transform(dev_data[features].values)

# Now extract target values 
y_train = train_data[target].values
y_dev = dev_data[target].values

# -------------------------------
# 4. Define Evaluation Metric 
# -------------------------------
def evaluate_model(y_true, y_pred): 
    """
    Evaluate the model using MSE and count the number of instances where the prediction is over/under the true value 
    """
    mse = mean_squared_error(y_true, y_pred) 
    under = np.sum(y_pred < y_true) 
    over = np.sum(y_pred > y_true) 
    return mse, under, over 

# -------------------------------
# 5. Model-Centric Approaches 
# -------------------------------
# 5.1. Baseline Model 
model_baseline = LinearRegression() 

baseline_start = time.time()                    # Start timer for baseline model training 
model_baseline.fit(X_train, y_train) 
baseline_end = time.time() - baseline_start     # Measure baseline model training time 

pred_baseline = model_baseline.predict(X_dev) 
mse_base, under_base, over_base = evaluate_model(y_dev, pred_baseline) 
print("Baseline Model Evaluation:")
print(f"MSE: {mse_base:.2f}")
print("Under: ", under_base)
print("Over: ", over_base)
print(f"Training Time: {baseline_end:.2f} seconds.\n")

# 5.2. Linear Regression with removed outliers 
threshold = np.percentile(y_train, 90) 
mask = train_data[train_data[target] <= threshold] 

X_train_mask = imputer.transform(mask[features].values)
y_train_mask = mask[target].values

# Now train the model 
mask_model = LinearRegression() 

mask_start = time.time()                    # Start timer for model training
mask_model.fit(X_train_mask, y_train_mask) 
mask_end = time.time() - mask_start         # Measure model training time 

pred_mask = mask_model.predict(X_dev) 
mse_mask, under_mask, over_mask = evaluate_model(y_dev, pred_mask) 
print("Linear Regression Model w/o Top 10% Outliers:")
print(f"MSE: {mse_mask:.2f}")
print("Under: ", under_mask)
print("Over: ", over_mask)
print(f"Training time: {mask_end:.2f} seconds.\n")

# 5.3. Log transformation of target variable 
y_train_log = np.log2(y_train + 1) 
model_log = LinearRegression() 

log_start = time.time()                     # Start timer for model training
model_log.fit(X_train, y_train_log) 
log_end = time.time() - log_start           # Measure model training time 

pred_log = np.power(2, model_log.predict(X_dev)) - 1
mse_log, under_log, over_log = evaluate_model(y_dev, pred_log) 
print("Model with Log Transformation of Target:")
print(f"MSE: {mse_log:.2f}")
print("Under: ", under_log)
print("Over: ", over_log)
print(f"Training time: {log_end:.2f} seconds.\n")

# 5.4. Pipeline with StandardScaler 
pipeline = Pipeline([ 
    ('imputer', SimpleImputer(strategy = 'mean')), 
    ('scaler', StandardScaler()), 
    ('regressor', LinearRegression())
])

# Fit the pipeline on the training data 
pipe_start = time.time()                   # Start timer for pipeline training  
pipeline.fit(train_data[features].values, y_train)
pipe_end = time.time() - pipe_start        # Measure pipeline training time 

# Predict of the development data 
pred_pipeline = pipeline.predict(dev_data[features].values)
mse_pipeline, under_pipeline, over_pipeline = evaluate_model(y_dev, pred_pipeline) 
print("Pipeline Model w Imputer and StandardScaler:")
print(f"MSE: {mse_pipeline:.2f}")
print("Under: ", under_pipeline)
print("Over: ", over_pipeline)
print(f"Training time: {pipe_end:.2f} seconds. \n")

# 5.5. ElasticNet with GridSeearchCV 
# Set up the pipeline with ElasticNet
pipeline_en = Pipeline([ 
    ('imputer', SimpleImputer(strategy = 'mean')), 
    ('scaler', StandardScaler()), 
    ('regressor', ElasticNet(max_iter = 10000))
])

# Set up the parameter grid for EN 
param_grid = {
    'regressor__alpha': [0.1, 1.0, 10.0], 
    'regressor__l1_ratio': [0.2, 0.5, 0.8] # Try 0.1, 0.5, 0.9 
}

# Set up the grid search 
grid_search = GridSearchCV(pipeline_en, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

# Fit the grid search on the training data and take the time 
grid_start = time.time()
grid_search.fit(train_data[features].values, y_train)
grid_end = time.time() - grid_start
best_model = grid_search.best_estimator_ 

# Predict using the best model 
pred_en = best_model.predict(dev_data[features].values) 
mse_en, under_en, over_en = evaluate_model(y_dev, pred_en)
print("ElasticNet Model with Hyperparameter Tuning:")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"MSE: {mse_en:.2f}")
print("Under: ", under_en)
print("Over: ", over_en)
print(f"Training time: {grid_end:.2f} seconds.\n")

# -------------------------------
# 6. Saving and Testing the Best Pipeline 
# -------------------------------
# Define the output directory and the full path for the saved pipeline.
output_dir = "/deac/csc/classes/csc373/santds21/assignment_5/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
pipeline_path = os.path.join(output_dir, "best_pipeline.pkl")

# Save the model pipeline.
with open("best_pipeline.pkl", "wb") as f:
    pickle.dump(model_log, f)

# Later, load the saved pipeline and test it on the development dataset.
with open("best_pipeline.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Predict using the loaded model (remember to invert the log-transform)
pred_start = time.time() 
pred_loaded = np.power(2, loaded_model.predict(X_dev)) - 1
pred_end = time.time() - pred_start
mse_loaded, under_loaded, over_loaded = evaluate_model(y_dev, pred_loaded)
print("Loaded Pipeline Evaluation:")
print(f"MSE: {mse_loaded:.2f}")
print("Under:", under_loaded)
print("Over:", over_loaded)
print(f"Training time: {pred_end:.2f} seconds.\n")
