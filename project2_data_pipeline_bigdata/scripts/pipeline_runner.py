from scripts.data_ingestion import ingest_log_data, create_spark_session
from scripts.silver_layer import process_silver_layer
from scripts.gold_layer import process_gold_layer
import os

def run_pipeline():
    """Main pipeline runner"""
    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_path, "data", "raw", "*.csv")
    silver_path = os.path.join(base_path, "data", "silver")
    gold_path = os.path.join(base_path, "data", "gold")
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        print("Starting data pipeline...")
        
        # Step 1: Data Ingestion
        print("Step 1: Ingesting data...")
        raw_df = ingest_log_data(spark, raw_path)
        
        # Step 2: Silver Layer Processing
        print("Step 2: Processing silver layer...")
        silver_df = process_silver_layer(raw_path, silver_path)
        
        # Step 3: Gold Layer Processing
        print("Step 3: Processing gold layer...")
        process_gold_layer(silver_path, gold_path)
        
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    run_pipeline()