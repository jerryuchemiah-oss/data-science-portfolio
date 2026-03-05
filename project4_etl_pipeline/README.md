# 🔄 ETL Data Pipeline — Multi-Source Consolidation

**Author:** Uche Jeremiah Nzubechukwu  
**Stack:** Python · Pandas · SQLite · Matplotlib  

## Overview
Production-style ETL pipeline that extracts from 3 simulated sources (REST API, CRM CSV, Legacy DB), cleans and transforms the data, consolidates into a SQLite warehouse, and generates an automated analytics dashboard.

## Architecture
```
[API Sales] ──┐
[CRM CSV]   ──┤── EXTRACT → TRANSFORM → LOAD → WAREHOUSE → DASHBOARD
[Legacy DB] ──┘
```

## Features
- **Extract**: Simulates REST API, CSV export, and SQLite DB sources
- **Transform**: Email validation, duplicate removal, revenue imputation, IQR outlier capping, derived features
- **Load**: SQLite warehouse + sales data mart via SQL JOIN
- **Quality Report**: Tracks rows in/out, duplicates, imputed values
- **Dashboard**: 6-panel matplotlib analytics report (revenue trends, region, channel, hourly patterns)
- **Logging**: Full pipeline audit trail saved to `etl_output/etl.log`

## Quick Start
```bash
pip install numpy pandas matplotlib
python etl_pipeline.py
```

## Outputs
```
etl_output/
├── warehouse.db        # SQLite data warehouse
├── sales_mart.csv      # Consolidated export
├── etl_dashboard.png   # Analytics dashboard
└── etl.log             # Audit log
```

## Swap in Real Data
```python
# Replace extractors with real sources:
import yfinance           # market data
import sqlalchemy         # database
import requests           # REST API
import boto3              # AWS S3
```

## Scale-Up Path
- Airflow / Prefect for orchestration
- PostgreSQL instead of SQLite
- dbt for transformations
- Great Expectations for data quality checks
