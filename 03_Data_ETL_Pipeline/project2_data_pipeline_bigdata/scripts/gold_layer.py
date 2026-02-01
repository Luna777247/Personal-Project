from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from scripts.data_ingestion import create_spark_session

def process_gold_layer(input_path, output_path):
    """Process data for gold layer: aggregations and business logic"""
    spark = create_spark_session()
    
    # Read silver data
    df = spark.read.parquet(input_path)
    
    # User activity aggregation
    user_activity = df.groupBy("user_id") \
        .agg(
            count("event_id").alias("total_events"),
            countDistinct("event_type").alias("unique_event_types"),
            min("event_timestamp").alias("first_activity"),
            max("event_timestamp").alias("last_activity")
        )
    
    # Event type aggregation
    event_summary = df.groupBy("event_type") \
        .agg(
            count("event_id").alias("event_count"),
            countDistinct("user_id").alias("unique_users")
        )
    
    # Daily activity
    daily_activity = df \
        .withColumn("date", to_date(col("event_timestamp"))) \
        .groupBy("date") \
        .agg(
            count("event_id").alias("daily_events"),
            countDistinct("user_id").alias("daily_users")
        )
    
    # Save gold layer data
    user_activity.write.mode("overwrite").parquet(f"{output_path}/user_activity")
    event_summary.write.mode("overwrite").parquet(f"{output_path}/event_summary")
    daily_activity.write.mode("overwrite").parquet(f"{output_path}/daily_activity")
    
    spark.stop()