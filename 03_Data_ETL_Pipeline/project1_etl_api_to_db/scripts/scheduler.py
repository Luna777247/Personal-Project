#!/usr/bin/env python3
import subprocess
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.etl_pipeline import run_etl

def main():
    """Scheduler script for daily ETL run"""
    try:
        print("Starting scheduled ETL pipeline...")
        run_etl()
        print("Scheduled ETL pipeline completed successfully")
    except Exception as e:
        print(f"Error in scheduled ETL: {e}")
        # Log error or send notification

if __name__ == "__main__":
    main()