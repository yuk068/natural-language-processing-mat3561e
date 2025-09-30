#!/usr/bin/env python3
"""
NLP Pipeline (PySpark) - Python implementation of Lab17_NLPPipeline.scala

Usage examples:
  # Default run (RegexTokenizer, HashingTF numFeatures=20000)
  python pipeline.py --data data/c4-train.00000-of-01024-30K.json

  # Use whitespace Tokenizer and smaller HashingTF
  python pipeline.py --tokenizer simple --num-features 1000

  # Use Word2Vec instead of HashingTF+IDF
  python pipeline.py --use-word2vec --word2vec-size 100 --word2vec-min-count 2

  # Add a toy LogisticRegression stage (requires label creation; script creates a synthetic label)
  python pipeline.py --add-logreg

Requirements:
  - PySpark (tested with 3.5.1)
  - loguru (optional; falls back to logging)
  - numpy, pandas (optional, for inspections)
"""

import os
import time
from datetime import datetime
import argparse
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    Word2Vec,
)
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import explode, col, length, when
from pyspark.sql.types import StringType

# Logging: try loguru, else fallback
try:
    from loguru import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("pipeline")

# Default paths
DEFAULT_RESULTS_DIR = "results"
DEFAULT_LOG_DIR = "log"
DEFAULT_RESULTS_FILE = os.path.join(DEFAULT_RESULTS_DIR, "pipeline_output.txt")
DEFAULT_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "metrics.log")


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def detect_text_column(df):
    """Return the name of a likely text column.
    Prioritize 'text', else choose the first string-typed column."""
    cols = df.columns
    if "text" in cols:
        return "text"
    # fallback: first column with string type
    for f in df.schema.fields:
        if isinstance(f.dataType, StringType):
            return f.name
    # otherwise, take first column (will be cast later)
    return cols[0] if cols else None


def build_pipeline(tokenizer_type: str, hashing_num_features: int, use_word2vec: bool, word2vec_size: int):
    stages = []
    # Tokenizer stage
    if tokenizer_type == "regex":
        tokenizer = RegexTokenizer(
            pattern=r"\s+|[.,;!?()\"'—\-:]",  # reasonable split
            inputCol="text",
            outputCol="tokens",
            toLowercase=True,
        )
    else:
        tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    stages.append(tokenizer)

    # Stop words removal
    stop_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    stages.append(stop_remover)

    # Vectorization
    if use_word2vec:
        w2v = Word2Vec(
            vectorSize=word2vec_size,
            minCount=1,
            inputCol="filtered_tokens",
            outputCol="features",
        )
        stages.append(w2v)
    else:
        hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=hashing_num_features)
        idf = IDF(inputCol="raw_features", outputCol="features")
        stages.append(hashing_tf)
        stages.append(idf)

    return Pipeline(stages=stages)


def synthetic_labeling(df, text_col="text"):
    """Create a simple synthetic binary label column for demonstration.
       Heuristic: label=1 if the text contains the token 'science' (case-insensitive),
       else 0. This is only for demonstrating LogisticRegression pipeline stage.
    """
    # Use simple contains check
    labeled = df.withColumn("label", when(col(text_col).rlike("(?i)\\bscience\\b"), 1.0).otherwise(0.0))
    return labeled


def write_results_file(result_path: str, rows, text_col="text"):
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"--- NLP Pipeline Output (First {len(rows)} results) ---\n")
        f.write(f"Output file generated at: {Path(result_path).absolute()}\n\n")
        for i, row in enumerate(rows):
            f.write("=" * 80 + "\n")
            txt = row[text_col] if row[text_col] is not None else ""
            preview = txt[:200].replace("\n", " ")
            f.write(f"[{i+1}] Original Text (first 200 chars): {preview}\n\n")
            feat = row["features"]
            f.write(f"Feature Vector: {feat}\n")
            f.write("\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="NLP Pipeline (PySpark) - Python implementation")
    parser.add_argument("--data", type=str, required=True, help="Path to gzipped JSON file (C4).")
    parser.add_argument("--limit", type=int, default=1000, help="Limit number of records for fast runs.")
    parser.add_argument("--tokenizer", type=str, choices=["regex", "simple"], default="regex", help="Tokenizer type.")
    parser.add_argument("--num-features", type=int, default=20000, help="HashingTF numFeatures (if used).")
    parser.add_argument("--use-word2vec", action="store_true", help="Use Word2Vec instead of HashingTF+IDF.")
    parser.add_argument("--word2vec-size", type=int, default=100, help="Word2Vec vector size.")
    parser.add_argument("--add-logreg", action="store_true", help="Add a LogisticRegression stage (requires 'label'). Script creates a synthetic label for demo.")
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS_FILE, help="Path to write results.")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG_FILE, help="Path to write metrics log.")
    parser.add_argument("--n-output", type=int, default=20, help="How many transformed rows to save to results file.")
    args = parser.parse_args()

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(args.results)
    log_path = Path(args.log)

    if results_path.name == Path(DEFAULT_RESULTS_FILE).name:
        args.results = str(results_path.with_name(f"pipeline_output_{ts_str}.txt"))
    if log_path.name == Path(DEFAULT_LOG_FILE).name:
        args.log = str(log_path.with_name(f"metrics_{ts_str}.log"))

    ensure_dirs(Path(args.results).parent, Path(args.log).parent)

    # Start Spark
    spark = SparkSession.builder.appName("NLP_Pipeline_PySpark").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session started.")
    start_job_ts = time.time()

    # Read
    logger.info(f"Reading dataset: {args.data} (limit {args.limit})")
    df = spark.read.json(args.data)
    if args.limit:
        df = df.limit(args.limit)
    record_count = df.count()
    logger.info(f"Loaded {record_count} records.")

    # Detect text column
    text_col = detect_text_column(df)
    if text_col is None:
        logger.error("No columns found in dataset.")
        spark.stop()
        return
    if text_col != "text":
        # cast selected column to 'text' column for pipeline compatibility
        df = df.withColumnRenamed(text_col, "text")
        text_col = "text"
        logger.info(f"Using detected text column and renamed to 'text' for pipeline.")

    df = df.select("text")  # keep only text column (pipeline expects 'text')

    # Build pipeline
    pipeline = build_pipeline(tokenizer_type=args.tokenizer, hashing_num_features=args.num_features,
                              use_word2vec=args.use_word2vec, word2vec_size=args.word2vec_size)

    # Optionally add LogisticRegression to pipeline after feature creation by building new pipeline after fit
    # We'll fit the feature pipeline first, then optionally fit LogisticRegression on top of extracted features.
    logger.info("Fitting feature pipeline...")
    t0 = time.time()
    pipeline_model = pipeline.fit(df)
    fit_time = time.time() - t0
    logger.info(f"Feature pipeline fitted in {fit_time:.2f}s")

    logger.info("Transforming dataset...")
    t1 = time.time()
    transformed = pipeline_model.transform(df).cache()
    # Force evaluation
    transformed_count = transformed.count()
    transform_time = time.time() - t1
    logger.info(f"Transformed {transformed_count} rows in {transform_time:.2f}s")

    # Compute vocabulary size after tokenization & stop words removal (distinct filtered tokens)
    try:
        vocab_df = transformed.select(explode(col("filtered_tokens")).alias("token")).filter(length(col("token")) > 1)
        actual_vocab_size = vocab_df.distinct().count()
    except Exception:
        actual_vocab_size = -1
    logger.info(f"Actual vocabulary size (distinct filtered tokens): {actual_vocab_size}")

    # If add-logreg: create synthetic labels if none, then train a logistic regression using 'features' and 'label'
    logreg_metrics = None
    if args.add_logreg:
        logger.info("Adding synthetic labels and training LogisticRegression (toy example)...")
        with_labels = synthetic_labeling(df, text_col="text")
        # Transform to extract features for labeled df
        labeled_transformed = pipeline_model.transform(with_labels).select("features", "label").cache()
        # Train logistic regression
        lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
        t2 = time.time()
        lr_model = lr.fit(labeled_transformed)
        lr_fit_time = time.time() - t2
        logger.info(f"LogisticRegression trained in {lr_fit_time:.2f}s")
        # Optionally evaluate on the same set (toy)
        pred = lr_model.transform(labeled_transformed)
        tp = pred.filter("label = prediction AND label = 1.0").count()
        tn = pred.filter("label = prediction AND label = 0.0").count()
        total = pred.count()
        acc = (tp + tn) / total if total > 0 else 0.0
        logreg_metrics = {"accuracy": acc, "total": total}
        logger.info(f"Toy LR accuracy on same data: {acc:.4f} ({tp+tn}/{total})")

    # Save sample results
    n_out = max(1, args.n_output)
    sample_rows = transformed.select("text", "features").take(n_out)
    write_results_file(args.results, sample_rows, text_col="text")
    logger.info(f"Wrote {len(sample_rows)} transformed example(s) to {args.results}")

    # Write metrics log
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If user gave a custom log filename, use it as prefix
    log_path = Path(args.log)
    if log_path.suffix:
        # insert timestamp before extension
        log_file = log_path.with_name(f"{log_path.stem}_{ts}{log_path.suffix}")
    else:
        # default: append .log and timestamp
        log_file = log_path.with_name(f"{log_path.name}_{ts}.log")

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write("--- NLP Pipeline Metrics ---\n")
        lf.write(f"Run timestamp: {ts}\n")
        lf.write(f"Data file: {args.data}\n")
        lf.write(f"Records processed: {record_count}\n")
        lf.write(f"Feature pipeline fit time (s): {fit_time:.4f}\n")
        lf.write(f"Data transform time (s): {transform_time:.4f}\n")
        lf.write(f"Actual vocab size (after preprocessing): {actual_vocab_size}\n")
        lf.write(f"Tokenizer: {args.tokenizer}\n")
        lf.write(f"HashingTF numFeatures (if used): {args.num_features}\n")
        lf.write(f"Used Word2Vec: {args.use_word2vec}\n")
        lf.write(f"Word2Vec size (if used): {args.word2vec_size}\n")
        if logreg_metrics:
            lf.write(f"LogisticRegression metrics: {logreg_metrics}\n")
        lf.write(f"Results file: {Path(args.results).absolute()}\n")
        lf.write("\nNote: If numFeatures < vocab size, hash collisions are expected.\n")

    logger.info(f"Wrote metrics to {log_file}")

    total_duration = time.time() - start_job_ts
    logger.info(f"Job completed in {total_duration:.2f} seconds")
    spark.stop()
    logger.info("Spark session stopped.")


if __name__ == "__main__":
    main()
