# Project 2: Mini Big Data Pipeline Simulation

## Overview
This project simulates a Big Data pipeline using Apache Spark in local mode. It ingests log data, processes it through silver and gold layers, and creates visualizations in Power BI. The project demonstrates fundamental Data Engineering concepts.

## Features
- Log data ingestion and processing
- Silver layer: Cleaned and transformed data
- Gold layer: Aggregated and business-ready data
- Power BI dashboard for visualization
- Local Spark processing

## Technologies
- Apache Spark (PySpark)
- Python 3.8+
- Pandas for data manipulation
- Power BI for visualization

## Project Structure
```
project2_data_pipeline_bigdata/
├── scripts/
│   ├── data_ingestion.py        # Log data ingestion
│   ├── silver_layer.py          # Data cleaning and transformation
│   ├── gold_layer.py            # Data aggregation
│   └── pipeline_runner.py       # Main pipeline orchestrator
├── data/
│   ├── raw/                     # Raw log files
│   ├── silver/                  # Processed data
│   └── gold/                    # Aggregated data
├── reports/
│   └── powerbi_dashboard.pbix   # Power BI file
├── requirements.txt
└── README.md
```

## Data Flow
1. **Raw Layer**: Log files ingested from various sources
2. **Silver Layer**: Data cleaning, deduplication, schema validation
3. **Gold Layer**: Business logic, aggregations, KPIs calculation

## Setup Instructions
1. Install Apache Spark and set SPARK_HOME
2. Install dependencies: `pip install -r requirements.txt`
3. Place log files in data/raw/
4. Run pipeline: `python scripts/pipeline_runner.py`
5. Open reports/powerbi_dashboard.pbix in Power BI

## Results
- Efficient log processing with Spark
- Clean data layers for different use cases
- Interactive Power BI visualizations
- Understanding of Big Data pipeline concepts