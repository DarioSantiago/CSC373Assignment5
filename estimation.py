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
Special thanks to Dr. Khuri for guidance and for providing the dataset.

Author: Dario Santiago Lopez, Anthony Roca, and ChatGPT
Date: April 2, 2025
"""


import os
import json
import gzip
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle
import time

# Configuration
DATA_PATH = "/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz"
OUTPUT_DIR = "/deac/csc/classes/csc373/rocaaj21/assignment_5/output"
RANDOM_STATE = 42

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------
# Utility Functions
# ----------------------------------
def load_data(path):
    with gzip.open(path) as f:
        data = [eval(l) for l in f]
    return pd.DataFrame(data)

def preprocess_data(df):
    df['review_word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['early_access'] = df['early_access'].astype(int)
    df['year'] = pd.to_datetime(df['date']).dt.year
    return df

def split_data(df, target='hours', split_ratio=0.8):
    split_idx = int(len(df) * split_ratio)
    train = df.iloc[:split_idx].copy()
    dev = df.iloc[split_idx:].copy()
    train = train.dropna(subset=[target])
    dev = dev.dropna(subset=[target])
    return train, dev

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    under = np.sum(y_pred < y_true)
    over = np.sum(y_pred > y_true)
    return mse, under, over

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def log_results(path, results):
    with open(path, "w") as f:
        for line in results:
            f.write(line + "\n")

# ----------------------------------
# Main Experiment
# ----------------------------------
def run_estimation():
    print("Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    train, dev = split_data(df)

    features = ['review_word_count', 'products', 'page_order', 'early_access', 'page']
    target = 'hours'

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(train[features].values)
    X_dev = imputer.transform(dev[features].values)
    y_train = train[target].values
    y_dev = dev[target].values

    results = ["Estimation Task Results", "======================="]

    # Baseline
    start = time.time()
    model_baseline = LinearRegression()
    model_baseline.fit(X_train, y_train)
    train_time = time.time() - start
    pred = model_baseline.predict(X_dev)
    mse, under, over = evaluate_model(y_dev, pred)
    results.append(f"Baseline MSE: {mse:.2f}, Under: {under}, Over: {over}, Train Time: {train_time:.2f}s")

    # Outlier removal
    threshold = np.percentile(y_train, 90)
    mask = train[train[target] <= threshold]
    X_train_mask = imputer.transform(mask[features].values)
    y_train_mask = mask[target].values
    start = time.time()
    model_mask = LinearRegression()
    model_mask.fit(X_train_mask, y_train_mask)
    train_time = time.time() - start
    pred = model_mask.predict(X_dev)
    mse, under, over = evaluate_model(y_dev, pred)
    results.append(f"Outlier-Removed MSE: {mse:.2f}, Under: {under}, Over: {over}, Train Time: {train_time:.2f}s")

    # Log-transformed target
    y_train_log = np.log2(y_train + 1)
    start = time.time()
    model_log = LinearRegression()
    model_log.fit(X_train, y_train_log)
    train_time = time.time() - start
    pred = np.power(2, model_log.predict(X_dev)) - 1
    mse, under, over = evaluate_model(y_dev, pred)
    results.append(f"Log-Transformed MSE: {mse:.2f}, Under: {under}, Over: {over}, Train Time: {train_time:.2f}s")

    # ElasticNet with GridSearch
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('regressor', ElasticNet(max_iter=10000, random_state=RANDOM_STATE))
    ])
    param_grid = {
        'regressor__alpha': [0.1, 1.0, 10.0],
        'regressor__l1_ratio': [0.2, 0.5, 0.8]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    start = time.time()
    grid.fit(train[features].values, y_train)
    train_time = time.time() - start
    best_model = grid.best_estimator_
    pred = best_model.predict(dev[features].values)
    mse, under, over = evaluate_model(y_dev, pred)
    results.append(f"ElasticNet MSE: {mse:.2f}, Under: {under}, Over: {over}, Params: {grid.best_params_}, Train Time: {train_time:.2f}s")

    # Save the best model (ElasticNet)
    model_path = os.path.join(OUTPUT_DIR, "best_pipeline.pkl")
    save_model(best_model, model_path)

    # Write results
    report_path = os.path.join(OUTPUT_DIR, "estimation_results.txt")
    log_results(report_path, results)
    print("Results written to", report_path)

if __name__ == "__main__":
    run_estimation()