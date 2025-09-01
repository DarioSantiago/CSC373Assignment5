Steam Game Hours Recommendation Pipeline

📌 Overview

This project implements a collaborative filtering recommendation system using PySpark's ALS (Alternating Least Squares) algorithm to predict log-transformed hours played for a user based on the preferences of other, similar users.
It was developed as part of CSC-373 (Data Mining) coursework and designed to handle large-scale datasets in a distributed computing environment.

The pipeline trains on Steam review data to produce recommendations while managing memory efficiency, data shuffling, and parallel computation through Spark configuration tuning.

📂 Project Structure

assignment_5/
│
├── scripts/
│   ├── recommendation.py         # Main PySpark training & evaluation script
│   ├── estimation.py              # Model comparison & selection
│   ├── classification.py          # Classification models (bonus/other task)
│   ├── code_segments.py           # Reusable helper functions/snippets
│   ├── steam_reviews_valid.json   # Input dataset (Steam reviews)
│   ├── best_pipeline.pkl          # Saved best regression pipeline (local ML)
│   └── best_recommendation_pipeline/ # Saved PySpark ALS recommendation model
│
├── output/
│   ├── best_pipeline.pkl
│   ├── best_recommendation_pipeline/
│   └── model_metrics.txt          # MSE, over/underprediction counts
│
└── README.md

⚙️ Features

PySpark ALS Recommendation Pipeline: Collaborative filtering using matrix factorization.

Log Transformation: Predicts log_hours to stabilize variance and improve accuracy.

Memory & Shuffle Optimization: Custom Spark configuration for large dataset handling.

Over/Under Prediction Tracking: Counts of predictions above and below actual hours.

Model Persistence: Saves trained pipelines for re-use without retraining.

Local & Cluster Execution: Runs on both local machines and HPC environments.

📊 Results (Example from HPC Run)

Metric	Value
MSE (log_hours)	6.60
Overpredictions (count)	28,700
Underpredictions (count)	90,088
Training Time (Full Data)	~13 min

🚀 How to Run
1️⃣ Local Execution
python scripts/recommendation.py

2️⃣ Cluster Execution (SLURM)
sbatch run_recommendation.slurm

💾 Saving & Loading Models

PySpark Recommendation Pipeline

model.write().overwrite().save("output/best_recommendation_pipeline")


scikit-learn Regression Pipeline

import pickle
with open("output/best_pipeline.pkl", "wb") as f:
    pickle.dump(model_log, f)

📈 Skills & Tools Used

Languages: Python, PySpark, SQL

Libraries: PySpark MLlib, scikit-learn, NumPy, Pandas

Concepts: Collaborative Filtering, Matrix Factorization, Feature Engineering

Environments: Local Python, HPC Cluster (SLURM), Linux CLI

Performance Tuning: Spark memory allocation, shuffle partitions, data repartitioning

🏆 Academic Context

This project was completed as part of Wake Forest University’s CSC-373: Data Mining course, focusing on machine learning for large-scale data and distributed systems.
It demonstrates the ability to:

Work with massive datasets efficiently.

Apply ML algorithms in distributed environments.

Tune computation for HPC cluster performance.
