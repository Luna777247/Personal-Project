from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from scripts.data_ingestion import create_spark_session

def process_silver_layer(input_path, output_path):
    """Process data for silver layer: cleaning and transformation"""
    spark = create_spark_session()
    
    # Read raw data
    df = spark.read.parquet(input_path)
    
    # Data cleaning
    df_clean = df \
        .dropDuplicates() \
        .filter(col("user_id").isNotNull()) \
        .withColumn("event_timestamp", to_timestamp(col("timestamp"))) \
        .withColumn("processed_at", current_timestamp())
    
    # Save to silver layer
    df_clean.write.mode("overwrite").parquet(output_path)
    
    spark.stop()
    return df_clean