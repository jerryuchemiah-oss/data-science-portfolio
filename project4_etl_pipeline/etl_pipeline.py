"""
╔══════════════════════════════════════════════════════════════╗
║   ETL Data Pipeline — Multi-Source Consolidation             ║
║   Author: Uche Jeremiah Nzubechukwu                          ║
║   Stack:  Python · Pandas · SQLite · Matplotlib              ║
╚══════════════════════════════════════════════════════════════╝

A production-style ETL (Extract → Transform → Load) pipeline that:
  1. Extracts data from multiple simulated sources (API, CSV, DB)
  2. Transforms and cleans each dataset
  3. Joins and consolidates into a unified data warehouse table
  4. Loads results into SQLite + exports to CSV
  5. Generates an automated data quality report

Run:
    python etl_pipeline.py
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import os
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path('etl_output')
OUTPUT_DIR.mkdir(exist_ok=True)
DB_PATH    = OUTPUT_DIR / 'warehouse.db'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'etl.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('ETL')

np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT — Simulated data sources
# ─────────────────────────────────────────────────────────────────────────────

class DataExtractor:
    """Simulates extraction from REST API, CSV, and a legacy database."""

    @staticmethod
    def extract_api_sales(n=800) -> pd.DataFrame:
        """Simulate a REST API response for sales transactions."""
        log.info("Extracting sales data from API...")
        dates = pd.date_range('2024-01-01', periods=n, freq='H')
        regions = ['North', 'South', 'East', 'West', 'Online']
        products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
        df = pd.DataFrame({
            'transaction_id':  [f'TXN_{str(i).zfill(6)}' for i in range(n)],
            'timestamp':        dates,
            'region':           np.random.choice(regions, n),
            'product_code':     np.random.choice(products, n),
            'quantity':         np.random.randint(1, 20, n),
            'unit_price':       np.random.uniform(10, 500, n).round(2),
            'discount_pct':     np.random.choice([0, 5, 10, 15, 20], n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
            'channel':          np.random.choice(['Web', 'Mobile', 'In-Store'], n, p=[0.45, 0.35, 0.2]),
            'rep_id':           [f'REP_{str(np.random.randint(1, 30)).zfill(3)}' for _ in range(n)],
            # Inject ~5% dirty data
            'customer_email':   [f'user{i}@example.com' if np.random.random() > 0.05
                                 else np.random.choice(['invalid-email', '', None, 'N/A'])
                                 for i in range(n)],
            'revenue':          np.where(np.random.random(n) < 0.03, None,
                                        np.random.uniform(50, 5000, n).round(2)),
        })
        log.info(f"   ✅ API: {len(df)} records extracted")
        return df

    @staticmethod
    def extract_csv_customers(n=300) -> pd.DataFrame:
        """Simulate CSV export from a CRM system."""
        log.info("Extracting customer data from CRM CSV...")
        products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
        df = pd.DataFrame({
            'customer_id':    [f'CUST_{str(i).zfill(5)}' for i in range(n)],
            'customer_name':  [f'Customer_{i}' for i in range(n)],
            'segment':        np.random.choice(['Enterprise', 'SMB', 'Consumer'], n, p=[0.2, 0.35, 0.45]),
            'country':        np.random.choice(['Nigeria', 'UK', 'USA', 'Germany', 'Kenya'], n,
                                               p=[0.35, 0.20, 0.25, 0.10, 0.10]),
            'signup_date':    pd.date_range('2020-01-01', periods=n, freq='D'),
            'lifetime_value': np.random.exponential(2000, n).round(2),
            'preferred_product': np.random.choice(products, n),
            # Introduce duplicates
            'email':          [f'user{i % (n - 15)}@example.com' for i in range(n)],
            'is_active':      np.random.binomial(1, 0.82, n),
        })
        log.info(f"   ✅ CRM CSV: {len(df)} records extracted")
        return df

    @staticmethod
    def extract_db_products() -> pd.DataFrame:
        """Simulate extraction from a legacy SQLite product catalog."""
        log.info("Extracting product catalog from legacy DB...")
        df = pd.DataFrame({
            'product_code':    ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E'],
            'product_name':    ['DataSense Pro', 'CloudStream X', 'AIAnalyzer', 'SecureNet', 'InsightDash'],
            'category':        ['Analytics', 'Infrastructure', 'AI/ML', 'Security', 'BI'],
            'cost_price':      [45.00, 120.00, 200.00, 85.00, 60.00],
            'list_price':      [99.00, 249.00, 399.00, 179.00, 129.00],
            'stock_units':     [1500, 800, 450, 1200, 900],
            'supplier':        ['TechCorp', 'CloudVendor', 'AI_Systems', 'SecureVend', 'BITools'],
        })
        log.info(f"   ✅ Legacy DB: {len(df)} product records extracted")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORM — Cleaning, validation, enrichment
# ─────────────────────────────────────────────────────────────────────────────

class DataTransformer:

    def __init__(self):
        self.quality_report = {}

    def transform_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Transforming sales data...")
        original_len = len(df)
        report = {}

        # 1. Fix timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date']       = df['timestamp'].dt.date
        df['hour']       = df['timestamp'].dt.hour
        df['day_of_week']= df['timestamp'].dt.day_name()

        # 2. Email validation
        email_valid = df['customer_email'].str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', na=False)
        report['invalid_emails'] = int((~email_valid).sum())
        df['customer_email'] = df['customer_email'].where(email_valid, other=None)

        # 3. Revenue imputation
        null_revenue = df['revenue'].isna().sum()
        report['imputed_revenue'] = int(null_revenue)
        df['revenue'] = df['revenue'].fillna(df['quantity'] * df['unit_price'] * (1 - df['discount_pct'] / 100))

        # 4. Derived columns
        df['gross_revenue']  = df['quantity'] * df['unit_price']
        df['net_revenue']    = df['gross_revenue'] * (1 - df['discount_pct'] / 100)
        df['row_hash']       = df.apply(lambda r: hashlib.md5(str(r['transaction_id']).encode()).hexdigest()[:8], axis=1)

        # 5. Outlier capping (IQR)
        q1, q3 = df['unit_price'].quantile([0.01, 0.99])
        df['unit_price'] = df['unit_price'].clip(q1, q3)

        report['rows_in']  = original_len
        report['rows_out'] = len(df)
        self.quality_report['sales'] = report
        log.info(f"   ✅ Sales: {original_len} → {len(df)} rows | Imputed {null_revenue} revenue values")
        return df

    def transform_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Transforming customer data...")
        original_len = len(df)
        report = {}

        # Remove duplicates on email
        dupe_count = df.duplicated('email').sum()
        df = df.drop_duplicates('email', keep='first')
        report['duplicates_removed'] = int(dupe_count)

        # Segment encoding
        seg_map = {'Enterprise': 2, 'SMB': 1, 'Consumer': 0}
        df['segment_code'] = df['segment'].map(seg_map)

        # Customer age (days since signup)
        df['signup_date'] = pd.to_datetime(df['signup_date'])
        df['customer_age_days'] = (datetime.now() - df['signup_date']).dt.days

        # LTV buckets
        df['ltv_tier'] = pd.cut(df['lifetime_value'],
                                 bins=[0, 500, 2000, 5000, np.inf],
                                 labels=['Bronze', 'Silver', 'Gold', 'Platinum'])

        report['rows_in']  = original_len
        report['rows_out'] = len(df)
        self.quality_report['customers'] = report
        log.info(f"   ✅ Customers: {original_len} → {len(df)} rows | Removed {dupe_count} duplicates")
        return df

    def transform_products(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Transforming product catalog...")
        df['margin_pct']     = ((df['list_price'] - df['cost_price']) / df['list_price'] * 100).round(1)
        df['is_high_margin'] = df['margin_pct'] > 50
        df['price_tier']     = pd.cut(df['list_price'], bins=[0, 100, 200, 400],
                                       labels=['Budget', 'Mid', 'Premium'])
        log.info(f"   ✅ Products: {len(df)} rows enriched")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# LOAD — SQLite data warehouse
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(db_path)
        log.info(f"📦 Connected to warehouse: {db_path}")

    def load(self, df: pd.DataFrame, table: str, if_exists='replace'):
        df.to_sql(table, self.conn, if_exists=if_exists, index=False)
        count = pd.read_sql(f"SELECT COUNT(*) AS n FROM {table}", self.conn).iloc[0, 0]
        log.info(f"   ✅ Loaded {count} rows into '{table}'")

    def create_mart(self):
        """Create a unified sales data mart by joining all tables."""
        log.info("Building consolidated data mart...")
        query = """
        CREATE TABLE IF NOT EXISTS sales_mart AS
        SELECT
            s.transaction_id,
            s.timestamp,
            s.date,
            s.region,
            s.product_code,
            p.product_name,
            p.category,
            p.margin_pct,
            s.quantity,
            s.unit_price,
            s.discount_pct,
            s.gross_revenue,
            s.net_revenue,
            s.channel,
            s.day_of_week,
            s.hour
        FROM sales s
        LEFT JOIN products p ON s.product_code = p.product_code
        """
        self.conn.execute("DROP TABLE IF EXISTS sales_mart")
        self.conn.execute(query)
        self.conn.commit()
        count = pd.read_sql("SELECT COUNT(*) AS n FROM sales_mart", self.conn).iloc[0, 0]
        log.info(f"   ✅ sales_mart created: {count} rows")
        return pd.read_sql("SELECT * FROM sales_mart", self.conn)

    def close(self):
        self.conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING — Automated data quality & analytics report
# ─────────────────────────────────────────────────────────────────────────────

def generate_dashboard(mart_df: pd.DataFrame, quality_report: dict):
    log.info("Generating analytics dashboard...")
    mart_df['date'] = pd.to_datetime(mart_df['date'])

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle('ETL Pipeline — Sales Analytics Dashboard', fontsize=16, fontweight='bold', y=0.98)

    # 1. Daily Revenue Trend
    ax1 = fig.add_subplot(gs[0, :2])
    daily = mart_df.groupby('date')['net_revenue'].sum().reset_index()
    daily['7d_avg'] = daily['net_revenue'].rolling(7).mean()
    ax1.fill_between(daily['date'], daily['net_revenue'], alpha=0.25, color='#2E5FA3')
    ax1.plot(daily['date'], daily['net_revenue'], color='#2E5FA3', lw=1, alpha=0.6, label='Daily Revenue')
    ax1.plot(daily['date'], daily['7d_avg'], color='#E74C3C', lw=2, label='7-Day Avg')
    ax1.set_title('Daily Net Revenue Trend', fontweight='bold')
    ax1.set_ylabel('Revenue ($)')
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

    # 2. Revenue by Region
    ax2 = fig.add_subplot(gs[0, 2])
    reg = mart_df.groupby('region')['net_revenue'].sum().sort_values()
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(reg)))
    ax2.barh(reg.index, reg.values, color=colors)
    ax2.set_title('Revenue by Region', fontweight='bold')
    ax2.set_xlabel('Revenue ($)')

    # 3. Revenue by Product
    ax3 = fig.add_subplot(gs[1, :2])
    prod = mart_df.groupby(['product_name'])['net_revenue'].sum().sort_values(ascending=False)
    ax3.bar(prod.index, prod.values, color=['#1B3A6B', '#2E5FA3', '#4A7FC1', '#6A9FD1', '#8ABDE0'])
    ax3.set_title('Revenue by Product', fontweight='bold')
    ax3.set_ylabel('Revenue ($)')
    ax3.tick_params(axis='x', rotation=15)

    # 4. Channel Mix
    ax4 = fig.add_subplot(gs[1, 2])
    ch = mart_df.groupby('channel')['net_revenue'].sum()
    ax4.pie(ch.values, labels=ch.index, autopct='%1.1f%%', startangle=90,
            colors=['#27AE60', '#2E5FA3', '#E74C3C'])
    ax4.set_title('Revenue by Channel', fontweight='bold')

    # 5. Hourly Revenue Heatmap-style
    ax5 = fig.add_subplot(gs[2, :2])
    hourly = mart_df.groupby('hour')['net_revenue'].sum()
    bar_colors = ['#E74C3C' if v == hourly.max() else '#2E5FA3' for v in hourly.values]
    ax5.bar(hourly.index, hourly.values, color=bar_colors, alpha=0.85)
    ax5.set_title('Revenue by Hour of Day (Peak Hour in Red)', fontweight='bold')
    ax5.set_xlabel('Hour')
    ax5.set_ylabel('Revenue ($)')
    ax5.set_xticks(range(0, 24))

    # 6. Data Quality Summary
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    qr_text = "DATA QUALITY REPORT\n" + "─" * 28 + "\n"
    for source, metrics in quality_report.items():
        qr_text += f"\n[{source.upper()}]\n"
        for k, v in metrics.items():
            qr_text += f"  {k}: {v}\n"
    ax6.text(0.05, 0.95, qr_text, transform=ax6.transAxes, fontsize=8.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#EEF2F7', alpha=0.8))

    plt.savefig(OUTPUT_DIR / 'etl_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("   ✅ Dashboard saved: etl_output/etl_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    start = datetime.now()
    log.info("=" * 60)
    log.info("  ETL PIPELINE STARTING")
    log.info("=" * 60)

    # EXTRACT
    extractor   = DataExtractor()
    raw_sales   = extractor.extract_api_sales()
    raw_customers = extractor.extract_csv_customers()
    raw_products  = extractor.extract_db_products()

    # TRANSFORM
    transformer   = DataTransformer()
    clean_sales     = transformer.transform_sales(raw_sales)
    clean_customers = transformer.transform_customers(raw_customers)
    clean_products  = transformer.transform_products(raw_products)

    # LOAD
    loader = DataLoader(DB_PATH)
    loader.load(clean_sales, 'sales')
    loader.load(clean_customers, 'customers')
    loader.load(clean_products, 'products')
    mart = loader.create_mart()

    # Export to CSV
    mart.to_csv(OUTPUT_DIR / 'sales_mart.csv', index=False)
    log.info(f"   ✅ CSV export: etl_output/sales_mart.csv")

    # Dashboard
    generate_dashboard(mart, transformer.quality_report)

    loader.close()

    elapsed = (datetime.now() - start).total_seconds()
    log.info("=" * 60)
    log.info(f"  ✅ PIPELINE COMPLETE in {elapsed:.1f}s")
    log.info(f"  📦 Output: {OUTPUT_DIR.resolve()}")
    log.info("=" * 60)

    # Print summary
    print(f"\n{'='*55}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*55}")
    print(f"  Sales transactions : {len(clean_sales):,}")
    print(f"  Unique customers   : {len(clean_customers):,}")
    print(f"  Products catalogued: {len(clean_products)}")
    print(f"  Mart rows          : {len(mart):,}")
    print(f"  Total revenue      : ${mart['net_revenue'].sum():,.2f}")
    print(f"  Top region         : {mart.groupby('region')['net_revenue'].sum().idxmax()}")
    print(f"  Top product        : {mart.groupby('product_name')['net_revenue'].sum().idxmax()}")
    print(f"  Elapsed time       : {elapsed:.1f}s")
    print(f"{'='*55}")


if __name__ == '__main__':
    run_pipeline()
