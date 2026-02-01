from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os

def create_spark_session():
    """Create Spark session for local processing"""
    return SparkSession.builder \
        .appName("BigDataPipeline") \
        .master("local[*]") \
        .getOrCreate()

def ingest_log_data(spark, input_path):
    """Ingest log data from files"""
    # Read log files (assuming CSV format for simplicity)
    df = spark.read.option("header", "true").csv(input_path)
    
    # Add ingestion timestamp
    df = df.withColumn("ingestion_timestamp", current_timestamp())
    
    return df

def save_to_silver(df, output_path):
    """Save raw data to silver layer"""
    df.write.mode("overwrite").parquet(output_path)