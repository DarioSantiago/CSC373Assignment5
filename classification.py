"""
classification.py - Binary Classification of Game-Playing Time from Steam Reviews

This script classifies game-playing time into two categories—high (above median) and low (below median).
- Loads a large dataset of Steam reviews and preprocesses the data.
- Binarizes the 'hours' feature using the median value from the training split.
- Implements and evaluates a Dummy Classifier (baseline) and a Logistic Regression classifier.
- Reports classification accuracy as well as counts for overpredictions (predicting high when true is low) and underpredictions (predicting low when true is high).
- Conducts data-centric experiments by training and testing on different review year splits (<=2014 vs. >=2015).

Acknowledgements:
This project is part of a Computer Science course on Data Mining.
Special thanks to the CSC373 faculty for their guidance and for providing the dataset.

Author: Dario Santiago Lopez, Anthony Roca, and ChatGPT
Date: April 2, 2025
"""

import json
import gzip 
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import time 

# -------------------------------
# 1. Binarize Target Feature for Classification + Data Cleaning 
# -------------------------------
print("\nClassification.py")
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

# code to split the data (decided to use Pandas since it's more convenient for feature engineering)
split_idx = int(len(df) * 0.8) 
train_data = df.iloc[:split_idx].copy() 
dev_data = df.iloc[split_idx:].copy() 

# Extract year from the date (decided to use Pandas for better performance and error handling)
train_data['year'] = pd.to_datetime(train_data['date']).dt.year 
dev_data['year'] = pd.to_datetime(dev_data['date']).dt.year 

# Convert 'hours' to numeric (if not already) and coerce errors to NaN
train_data['hours'] = pd.to_numeric(train_data['hours'], errors = 'coerce')
dev_data['hours'] = pd.to_numeric(dev_data['hours'], errors = 'coerce')

# Drop rows where 'hours' is NaN so that median computation is valid
train_data = train_data.dropna(subset = ['hours'])
dev_data = dev_data.dropna(subset = ['hours'])

# Compute review word count as a feature 
train_data['review_word_count'] = train_data['text'].apply(lambda x: len(x.split()))
dev_data['review_word_count'] = dev_data['text'].apply(lambda x: len(x.split()))

# Compute median hours from the training data (from estimation.py split)
median_hours = np.median(train_data['hours'].values)
print("Median hours (training):", median_hours)

# Create a binary target: 1 if hours > median, else 0.
train_data['target_class'] = (train_data['hours'] > median_hours).astype(int)
dev_data['target_class'] = (dev_data['hours'] > median_hours).astype(int)

# Use the same features as before (non-string features only)
features = ['review_word_count', 'products', 'page_order', 'early_access', 'page']

# Define an evaluation function for classification
def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    # Overpredictions: predicted high (1) but true is low (0)
    overpred = np.sum((y_pred == 1) & (y_true == 0))
    # Underpredictions: predicted low (0) but true is high (1)
    underpred = np.sum((y_pred == 0) & (y_true == 1))
    return accuracy, overpred, underpred

# -------------------------------
# 2.0 Dummy Classifier Pipeline
# -------------------------------
# Dummy classifier to establish a baseline 
dummy_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'mean')),
    ('scaler', StandardScaler()),
    ('dummy', DummyClassifier(strategy = 'most_frequent'))
])
# Train the dummy classifier
dummy_start = time.time() # Start timer for dummy classifier training 
dummy_pipeline.fit(train_data[features].values, train_data['target_class'].values)
dummy_end = time.time() - dummy_start # Measure dummy classifier training time 

# Make predictions on the dev set
dummy_pred = dummy_pipeline.predict(dev_data[features].values)
dummy_accuracy, dummy_over, dummy_under = evaluate_classification(dev_data['target_class'].values, dummy_pred)

print("\nDummy Classifier Performance:")
print(f"Accuracy: {dummy_accuracy:.2f}")
print("Over:", dummy_over)
print("Under:", dummy_under)
print(f"Training Time: {dummy_end:.2f} seconds.\n")

# -------------------------------
# 2.1 Baseline Classifier Pipeline using Logistic Regression
# -------------------------------
baseline_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])
# Train the baseline model
base_start = time.time() # Start timer for baseline model training 
baseline_pipeline.fit(train_data[features].values, train_data['target_class'].values)
base_end = time.time() - base_start # Measure baseline model training time

# Make predictions on the dev set
baseline_pred = baseline_pipeline.predict(dev_data[features].values)
baseline_accuracy, baseline_over, baseline_under = evaluate_classification(dev_data['target_class'].values, baseline_pred)

print("Baseline Classifier (Logistic Regression) Performance:")
print(f"Accuracy: {baseline_accuracy:.2f}" )
print("Over:", baseline_over)
print("Under:", baseline_under)
print(f"Training Time: {base_end:.2f} seconds.\n")

#--------------------------------
# 2.2 Random Forest Classifier Pipeline (just for fun)
# -------------------------------- 
# rf_pipe = Pipeline([ 
#     ('imputer', SimpleImputer(strategy = 'mean')), 
#     ('scaler', StandardScaler()), 
#     ('classifier', RandomForestClassifier(n_estimators = 100, random_state = 42))
# ])

# # Train the Random Forest model 
# rf_start = time.time() # Start timer for RF model training 
# rf_pipe.fit(train_data[features].values, train_data['target_class'].values)
# rf_end = time.time() - rf_start # Measure RF model training time 

# # Make predictions on the dev set 
# rf_pred = rf_pipe.predict(dev_data[features].values)
# rf_accuracy, rf_over, rf_under = evaluate_classification(dev_data['target_class'].values, rf_pred)

# print("Random Forest Classifier Performance: ")
# print(f"Accuracy: {rf_accuracy:.2f}")
# print("Over: ", rf_over)
# print("Under: ", rf_under)
# print(f"Training Time: {rf_end:.2f} seconds. \n")

# -------------------------------
# 3.0 Data-Centric Experiments Based on Review Year
# -------------------------------
# Combine train and dev data for experiments 
all_data = pd.concat([train_data, dev_data], axis = 0, ignore_index = True) 
training_avg = median_hours # Use the median hours from the training data for consistency
all_data['target_class'] = (all_data['hours'] > training_avg).astype(int)

# Ensure necessary features exist; add review_word_count if not already present
if 'review_word_count' not in all_data.columns:
    all_data['review_word_count'] = all_data['text'].apply(lambda x: len(x.split()))

# Use the same feature set for experiments
features_exp = ['review_word_count', 'products', 'page_order', 'early_access', 'page']

# Experiment 1: Train on reviews from 2014 or earlier, test on reviews from 2015 or later.
train_2014 = all_data[all_data['year'] <= 2014]
test_2015 = all_data[all_data['year'] >= 2015]

exp1_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'mean')),
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression(max_iter = 1000))
])

# -------------------------------
# Debugging: Check the distribution
# print("Experiment 1: Distribution in training data (<= 2014):")
# print(train_2014['target_class'].value_counts())
# --------------------------------- 
# Train on reviews from 2014 or earlier, test on reviews from 2015 or later
exp1_start = time.time() # Start timer for experiment 1 
exp1_pipeline.fit(train_2014[features_exp].values, train_2014['target_class'].values)
exp1_end = time.time() - exp1_start # Measure experiment 1 training time

# Make predictions on the test set 
exp1_pred = exp1_pipeline.predict(test_2015[features_exp].values)
exp1_accuracy, exp1_over, exp1_under = evaluate_classification(test_2015['target_class'].values, exp1_pred)

print("Experiment 1: Train on reviews (<= 2014), test on reviews (>= 2015)")
print(f"Accuracy: {exp1_accuracy:.2f}")
print("Over:", exp1_over)
print("Under:", exp1_under)
print(f"Training Time: {exp1_end:.2f} seconds.\n")

# -------------------------------
# 3.2 Experiment 2: Train on reviews from 2015 or later, test on reviews from 2014 or earlier.
# -------------------------------
# Train on reviews from 2015 or later, test on reviews from 2014 or earlier 
train_2015 = all_data[all_data['year'] >= 2015]
test_2014 = all_data[all_data['year'] <= 2014]

exp2_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])
# Train on reviews from 2015 or later, test on reviews from 2014 or earlier 
exp2_start = time.time() # Start timer for exp2 
exp2_pipeline.fit(train_2015[features_exp].values, train_2015['target_class'].values)
exp2_end = time.time() - exp2_start # Measure exp2 training time

# Make predictions on the test set 
exp2_pred = exp2_pipeline.predict(test_2014[features_exp].values)
exp2_accuracy, exp2_over, exp2_under = evaluate_classification(test_2014['target_class'].values, exp2_pred)

print("Experiment 2: Train on reviews (>= 2015), test on reviews (<= 2014)")
print(f"Accuracy: {exp2_accuracy:.2f}" )
print("Over:", exp2_over)
print("Under:", exp2_under)
print(f"Training Time: {exp2_end:.2f} seconds.\n")
