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
Special thanks to the CSC373 faculty for their guidance and for providing the dataset.

Author: Dario Santiago Lopez, Anthony Roca, and ChatGPT
Date: April 2, 2025
""" 

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
import ast
import json
import gzip 
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log2, when, expr, count 
from pyspark.ml.recommendation import ALS 
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType 
import time 
import warnings 
warnings.filterwarnings("ignore")

# ------------------------------------
# 1. Create Spark Session and begin loading data 
# ------------------------------------
print("\nRecommendation.py")
print("--------------------")
print("Loading data...\n")

# Create spark session (orginially at 8g)
spark = SparkSession.builder.appName("RecommendationPipeline") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.broadcast.compress", "true") \
    .config("spark.ui.showConsoleProgress", "false") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR") # Used to get rid of warning messages

# Begin loading data and measure time and define file paths
input_path = "/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz"
output_path = "/deac/csc/classes/csc373/santds21/assignment_5/data/steam_reviews_valid.json"

data_start = time.time() # Measure time to load data 

# Preprocess the file if the valid JSON file does not exist
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

# Create schema to read json file
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

data = spark.read.schema(schema).json(output_path)
data_end = time.time() - data_start 
print(f"Data loaded in {data_end:.2f} seconds.")

# Convert hours to float (consider just keeping them integers as well) and filter out rows with missing hours 
data = data.withColumn("hours", col("hours").cast(FloatType()))
data = data.filter(col("hours").isNotNull())

# Create a new column for log-transformed hours: log2(hours + 1) 
data = data.withColumn("log_hours", log2(col("hours") + 1))

# Create a datasubset to deal with memory issues during training 
data_subset = 1.0
if data_subset < 1.0: 
    print(f"Using a subset of the data: {data_subset * 100:.0f}% of the total records.\n")
    data = data.sample(withReplacement = False, fraction = data_subset, seed = 42)

# ------------------------------------
# 2. Train/Development Split
# ------------------------------------
# Use randomSplit to randomly partition the data into 80% training and 20% development sets
train_data, dev_data = data.randomSplit([0.8, 0.2], seed=42)
train_data = train_data.repartition(100)

# Print the counts to verify the split
print("Training data count:", train_data.count())
print("Development data count:", dev_data.count())

# ------------------------------------
# 3. Indexing User and Product IDs
# ------------------------------------
# Convert 'username' and 'product_id' to numeric function
user_indexer = StringIndexer(inputCol = "username", outputCol = "userIndex", handleInvalid="skip")
product_indexer = StringIndexer(inputCol = "product_id", outputCol = "productIndex", handleInvalid="skip")

# ------------------------------------
# 4. ALS Model Setup 
# ------------------------------------
# Create ALS to fill in missing ratings 
als = ALS( 
    userCol = "userIndex", 
    itemCol = "productIndex", 
    ratingCol = "log_hours", 
    coldStartStrategy = "drop", # Drop predictions for unseen user/items 
    nonnegative = True, 
    maxIter = 10,               # Adjust for tuning
    regParam = 0.1,             # Adjust for tuning
    rank = 10                   # Adjust for tuning
)

# Build a Pipeline for indexing and ALS 
pipeline = Pipeline(stages = [user_indexer, product_indexer, als])

# ------------------------------------
# 5. Train Recommendation Model 
# ------------------------------------
time_start = time.time() # Measure how long it takes to train the model
model = pipeline.fit(train_data)
time_end = time.time() - time_start # Get total time 
print(f"Training Time: {time_end:.2f} seconds.")

# ------------------------------------
# 6. Make Predicions on the Dev Set 
# ------------------------------------
predictions = model.transform(dev_data)

# ------------------------------------
# 7. Evaulate Model 
# ------------------------------------
# Predictions DataFrame should now contain a "prediction" column (log_hours predicted)
evaluator = RegressionEvaluator( 
    labelCol = "log_hours", 
    predictionCol = "prediction", 
    metricName = "mse"
)
mse = evaluator.evaluate(predictions)
print(f"Recommendation Model MSE (log_hours): {mse:.2f}")
# Compute overpredictions and underpredictions on log_hours 
# Overprediction: prediction > true log_hours, Underprediction: prediction < true log_hours.
predictions = predictions.withColumn(
    "over_pred", when(col("prediction") > col("log_hours"), 1).otherwise(0)
).withColumn(
    "under_pred", when(col("prediction") < col("log_hours"), 1).otherwise(0)
)

over = predictions.groupBy().sum("over_pred").collect()[0][0]
under = predictions.groupBy().sum("under_pred").collect()[0][0]
print("Over (count): ", over)
print(f"Under (count): {under}\n")

# -------------------------------
# 8. Save the Pipeline Model
# -------------------------------
output_dir = "/deac/csc/classes/csc373/santds21/assignment_5/output/best_recommendation_pipeline"
model.write().overwrite().save(output_dir)

# Stop the Spark session
spark.stop()
