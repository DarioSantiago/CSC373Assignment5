"""
recommendation.py - Collaborative Filtering Recommendation for Log-Transformed Hours

This script uses PySpark to train a recommendation pipeline based on ALS.
It makes predictions of log-transformed hours played for a user by leveraging collaborative filtering.
- Loads and preprocesses the gzipped JSON data.
- Splits the data into training (80%) and development (20%) sets.
- Converts user and product IDs to numeric indices.
- Computes a new rating column: log2(hours + 1).
- Trains an ALS recommendation model on the training set.
- Evaluates predictions on the development set by reporting MSE and counts of overpredictions/underpredictions.
  
Acknowledgements:
This project is part of a Computer Science course on Data Mining.
Special thanks to Dr. Khuri for guidance and for providing the dataset.

Author: Dario Santiago Lopez, Anthony Roca, and ChatGPT
Date: April 2, 2025
""" 

import os
import ast
import json
import gzip
import time
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log2, when
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Configuration
INPUT_PATH = "/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz"
JSON_PATH = "/deac/csc/classes/csc373/rocaaj21/assignment_5/data/steam_reviews_valid.json"
OUTPUT_DIR = "/deac/csc/classes/csc373/rocaaj21/assignment_5/output/best_recommendation_pipeline"
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# Utility Functions
def preprocess_json(input_path, output_path):
    if not os.path.exists(output_path):
        print("Preprocessing file to create valid JSON...")
        with gzip.open(input_path, "rt", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                try:
                    record = ast.literal_eval(line.strip())
                    json_record = json.dumps(record)
                    fout.write(json_record + "\n")
                except Exception as e:
                    print("Error processing line:", e)
        print("Preprocessing complete.\n")

def create_spark_session():
    spark = SparkSession.builder.appName("RecommendationPipeline") \
        .config("spark.driver.memory", "16g") \
        .config("spark.executor.memory", "16g") \
        .config("spark.broadcast.compress", "true") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_and_prepare_data(spark, path):
    schema = StructType([
        StructField("username", StringType(), True),
        StructField("hours", FloatType(), True),
        StructField("products", IntegerType(), True),
        StructField("product_id", StringType(), True),
        StructField("page_order", IntegerType(), True),
        StructField("date", StringType(), True),
        StructField("text", StringType(), True),
        StructField("early_access", IntegerType(), True),
        StructField("page", IntegerType(), True)
    ])
    data = spark.read.schema(schema).json(path)
    data = data.withColumn("hours", col("hours").cast(FloatType()))
    data = data.filter(col("hours").isNotNull())
    data = data.withColumn("log_hours", log2(col("hours") + 1))
    return data

def evaluate_predictions(predictions):
    evaluator = RegressionEvaluator(labelCol="log_hours", predictionCol="prediction", metricName="mse")
    mse = evaluator.evaluate(predictions)
    predictions = predictions.withColumn("over_pred", when(col("prediction") > col("log_hours"), 1).otherwise(0))
    predictions = predictions.withColumn("under_pred", when(col("prediction") < col("log_hours"), 1).otherwise(0))
    over = predictions.groupBy().sum("over_pred").collect()[0][0]
    under = predictions.groupBy().sum("under_pred").collect()[0][0]
    return mse, over, under

def run_recommendation():
    print("\nRecommendation.py")
    print("--------------------")
    print("Loading data...\n")

    preprocess_json(INPUT_PATH, JSON_PATH)
    spark = create_spark_session()

    data_start = time.time()
    data = load_and_prepare_data(spark, JSON_PATH)
    data_subset = 0.2  # Use 20% of the data
    if data_subset < 1.0:
        print(f"Using a subset of the data: {data_subset * 100:.0f}%")
        data = data.sample(withReplacement=False, fraction=data_subset, seed=42)

    data_end = time.time() - data_start
    print(f"Data loaded in {data_end:.2f} seconds.")

    train_data, dev_data = data.randomSplit([0.8, 0.2], seed=42)
    train_data = train_data.repartition(100)
    print("Training data count:", train_data.count())
    print("Development data count:", dev_data.count())

    # --- Break apart indexing and ALS manually ---
    print("Indexing users and products...")

    user_indexer = StringIndexer(inputCol="username", outputCol="userIndex", handleInvalid="skip")
    product_indexer = StringIndexer(inputCol="product_id", outputCol="productIndex", handleInvalid="skip")

    user_indexer_model = user_indexer.fit(train_data)
    train_data_indexed = user_indexer_model.transform(train_data)

    product_indexer_model = product_indexer.fit(train_data_indexed)
    train_data_indexed = product_indexer_model.transform(train_data_indexed)

    # ALS setup
    als = ALS(
        userCol="userIndex",
        itemCol="productIndex",
        ratingCol="log_hours",
        coldStartStrategy="drop",
        nonnegative=True,
        maxIter=10,
        regParam=0.1,
        rank=10
    )

    print("Training ALS model...")
    start = time.time()
    als_model = als.fit(train_data_indexed)
    train_time = time.time() - start
    print(f"Training Time: {train_time:.2f} seconds.")

    dev_indexed = user_indexer_model.transform(dev_data)
    dev_indexed = product_indexer_model.transform(dev_indexed)

    predictions = als_model.transform(dev_indexed)

    mse, over, under = evaluate_predictions(predictions)
    print(f"Recommendation Model MSE (log_hours): {mse:.2f}")
    print("Over (count):", over)
    print("Under (count):", under)

    # Save components separately
    user_indexer_model.write().overwrite().save(os.path.join(OUTPUT_DIR, "user_indexer"))
    product_indexer_model.write().overwrite().save(os.path.join(OUTPUT_DIR, "product_indexer"))
    als_model.write().overwrite().save(os.path.join(OUTPUT_DIR, "als_model"))

    report_lines = [
        "Recommendation Task Results",
        "============================",
        f"MSE (log_hours): {mse:.2f}",
        f"Overpredicted: {over}",
        f"Underpredicted: {under}"
    ]
    with open(os.path.join(OUTPUT_DIR, "recommendation_results.txt"), "w") as f:
        for line in report_lines:
            print(line)
            f.write(line + "\n")

    spark.stop()

if __name__ == "__main__":
    run_recommendation()