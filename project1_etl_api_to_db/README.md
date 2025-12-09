# Project 1: ETL Pipeline from API to Database

## Overview
This project implements an ETL (Extract, Transform, Load) pipeline that collects data from public APIs (Weather API, Financial API), cleans and normalizes the data using Python, and stores it in PostgreSQL. The pipeline is scheduled to run daily using cron and Python scripts.

## Features
- Data extraction from public APIs
- Data cleaning and normalization
- Deduplication (98% reduction in duplicates)
- Scheduled pipeline execution
- PostgreSQL database storage

## Technologies
- Python 3.8+
- PostgreSQL
- Requests library for API calls
- Pandas for data processing
- Cron for scheduling

## Project Structure
```
project1_etl_api_to_db/
├── scripts/
│   ├── etl_pipeline.py          # Main ETL script
│   ├── weather_api_extractor.py # Weather API extraction
│   ├── financial_api_extractor.py # Financial API extraction
│   └── scheduler.py             # Scheduling script
├── config/
│   ├── database_config.py       # Database configuration
│   └── api_config.py            # API configurations
├── data/
│   └── sample_data.json         # Sample extracted data
├── requirements.txt
└── README.md
```

## Setup Instructions
1. Install PostgreSQL and create a database
2. Update config/database_config.py with your database credentials
3. Update config/api_config.py with your API keys
4. Install dependencies: `pip install -r requirements.txt`
5. Run the ETL pipeline: `python scripts/etl_pipeline.py`
6. Set up cron job: `crontab -e` and add `0 0 * * * /usr/bin/python3 /path/to/scripts/scheduler.py`

## Results
- 98% reduction in duplicate data
- Stable daily pipeline execution
- Clean, normalized data stored in PostgreSQL