"""
Lab 5: Task 3 - Spark Sentiment Analysis Example

This script builds and evaluates a text classification pipeline
using PySpark, as described in lab5_text_classification.pdf.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    # 1. Initialize Spark Session
    # Set log level to WARN to reduce verbosity
    spark = SparkSession.builder \
        .appName("SentimentAnalysis") \
        .config("spark.driver.extraJavaOptions", "-Dlog4j.level=WARN") \
        .config("spark.executor.extraJavaOptions", "-Dlog4j.level=WARN") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")

    # Define data path
    data_path = "data/sentiments.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please create this file (e.g., from the README) and try again.")
        spark.stop()
        return

    print(f"Reading data from {data_path}")

    # 2. Load Data
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    
    print("Data schema:")
    df.printSchema()

    # Drop rows with null sentiment values
    df = df.dropna(subset=["sentiment"])
    
    # Convert -1/1 labels to 0/1
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
    
    print("\nDataFrame with labels:")
    df.show(5, truncate=False)

    # 3. Build Preprocessing Pipeline
    
    # Stage 1: Tokenizer
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    
    # Stage 2: StopWordsRemover
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    # Stage 3: HashingTF
    # numFeatures=1000 means it will hash features into 1000 buckets
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
    
    # Stage 4: IDF
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    # 4. Train the Model
    
    # Stage 5: LogisticRegression
    lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")
    
    # Assemble the Pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])
    
    # Split the data
    (trainingData, testData) = df.randomSplit([0.8, 0.2], seed=42)
    
    print("Training model...")
    # Training: Call pipeline.fit() on the training data
    model = pipeline.fit(trainingData)
    print("Model trained. Evaluating...")

    # 5. Evaluate the Model
    
    # Use model.transform() on the test data to get predictions
    predictions = model.transform(testData)
    
    # Show some predictions
    predictions.select("text", "label", "prediction", "probability").show(5)

    # Evaluator for Accuracy
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator_acc.evaluate(predictions)
    
    # Evaluator for F1-score
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    f1 = evaluator_f1.evaluate(predictions)

    print(f"\nAccuracy: {accuracy}")
    print(f"F1-score: {f1}\n")
    
    # Stop the Spark session
    spark.stop()
    print("Spark session stopped.")

if __name__ == "__main__":
    main()
