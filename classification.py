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
Special thanks to Dr. Khuri for guidance and for providing the dataset.

Author: Dario Santiago Lopez, Anthony Roca, and ChatGPT
Date: April 2, 2025
"""


import os
import gzip
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Configuration
DATA_PATH = "/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz"
OUTPUT_DIR = "/deac/csc/classes/csc373/rocaaj21/assignment_5/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utility Functions
def load_data(path):
    with gzip.open(path) as f:
        return pd.DataFrame([eval(l) for l in f])

def preprocess_data(df):
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
    df = df.dropna(subset=['hours'])
    df['review_word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['early_access'] = df['early_access'].astype(int)
    df['year'] = pd.to_datetime(df['date']).dt.year
    return df

def split_data(df):
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].copy()
    dev = df.iloc[split_idx:].copy()
    return train, dev

def binarize_target(df, threshold):
    df['target_class'] = (df['hours'] > threshold).astype(int)
    return df

def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    overpred = np.sum((y_pred == 1) & (y_true == 0))
    underpred = np.sum((y_pred == 0) & (y_true == 1))
    return accuracy, overpred, underpred

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def run_classifier(pipeline, X_train, y_train, X_test, y_test, label):
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = pipeline.predict(X_test)
    acc, over, under = evaluate_classification(y_test, y_pred)
    return acc, f"{label} Accuracy: {acc:.2f}, Over: {over}, Under: {under}, Train Time: {train_time:.2f}s", pipeline

def run_classification():
    print("Loading and preprocessing data...")
    df = preprocess_data(load_data(DATA_PATH))
    train, dev = split_data(df)

    median_hours = np.median(train['hours'].values)
    print("Median hours (training):", median_hours)

    train = binarize_target(train, median_hours)
    dev = binarize_target(dev, median_hours)

    features = ['review_word_count', 'products', 'page_order', 'early_access', 'page']
    X_train = train[features].values
    y_train = train['target_class'].values
    X_dev = dev[features].values
    y_dev = dev['target_class'].values

    results = ["Classification Task Results", "============================"]

    # Dummy Classifier
    dummy_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        (('dummy', DummyClassifier(strategy='most_frequent', random_state=42)))
    ])
    acc, msg, _ = run_classifier(dummy_pipeline, X_train, y_train, X_dev, y_dev, "Dummy Classifier")
    results.append(msg)

    # Logistic Regression
    lr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        (('classifier', LogisticRegression(max_iter=1000, random_state=42)))
    ])
    acc_lr, msg, _ = run_classifier(lr_pipeline, X_train, y_train, X_dev, y_dev, "Logistic Regression")
    results.append(msg)

    # Save the best model (Logistic Regression)
    save_model(lr_pipeline, os.path.join(OUTPUT_DIR, "best_classifier.pkl"))

    # Data-centric experiments
    all_data = pd.concat([train, dev], ignore_index=True)
    all_data = binarize_target(all_data, median_hours)

    train_2014 = all_data[all_data['year'] <= 2014]
    test_2015 = all_data[all_data['year'] >= 2015]
    results.append(run_classifier(lr_pipeline,
                                  train_2014[features].values,
                                  train_2014['target_class'].values,
                                  test_2015[features].values,
                                  test_2015['target_class'].values,
                                  "Experiment 1 (<=2014 vs >=2015)")[1])

    train_2015 = all_data[all_data['year'] >= 2015]
    test_2014 = all_data[all_data['year'] <= 2014]
    results.append(run_classifier(lr_pipeline,
                                  train_2015[features].values,
                                  train_2015['target_class'].values,
                                  test_2014[features].values,
                                  test_2014['target_class'].values,
                                  "Experiment 2 (>=2015 vs <=2014)")[1])

    # Save results
    with open(os.path.join(OUTPUT_DIR, "classification_results.txt"), "w") as f:
        for line in results:
            print(line)
            f.write(line + "\n")

if __name__ == "__main__":
    run_classification()
