import pandas as pd
import os
import numpy as np
from sqlalchemy import create_engine
import tkinter as tk
from tkinter import ttk
import logging
from datetime import datetime as dt

# ---------- CONFIG ----------

FILENAME = 'SamplePrice_wETH_wBTC_wSOL.csv'
DOWNLOADS_PATH = os.path.expanduser('~/Downloads')
FILE_PATH = os.path.join(DOWNLOADS_PATH, FILENAME)

DB_NAME = 'crypto_data'
DB_USER = 'postgres'
DB_PASSWORD = 'yourpassword'  # <-- Change this securely
DB_HOST = 'localhost'
DB_PORT = 5432
TABLE_NAME = 'crypto_metrics'

HOURLY_TO_ANNUAL_VOL = np.sqrt(365.25 * 24)
ROLLING_WINDOWS = {'1d': 24, '7d': 168, '30d': 720, '90d': 2160}

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# ---------- CORE FUNCTIONS ----------

def compute_log_returns(df):
    df['log_return'] = df.groupby('symbol')['close'].transform(lambda x: np.log(x / x.shift(1)))
    return df

def compute_rolling_std(df, windows):
    for label, window in windows.items():
        df[f'rolling_std_{label}'] = df.groupby('symbol')['log_return'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        df[f'annual_rolling_std_{label}'] = df[f'rolling_std_{label}'] * HOURLY_TO_ANNUAL_VOL
    return df

def compute_period_std(df):
    df['period_std'] = df.groupby('symbol')['log_return'].transform('std')
    df['annual_std'] = df['period_std'] * HOURLY_TO_ANNUAL_VOL
    return df

def compute_rolling_correlation(df, symbol1, symbol2, window, label):
    sub = df[df['symbol'].isin([symbol1, symbol2])][['hour', 'symbol', 'log_return']].copy()
    pivot = sub.pivot(index='hour', columns='symbol', values='log_return').dropna()

    rolling_corr = pivot[symbol1].rolling(window=window, min_periods=window).corr(pivot[symbol2])
    corr_df = rolling_corr.reset_index(name=label)
    df = df.merge(corr_df, on='hour', how='left')
    df.loc[~df['symbol'].isin([symbol1, symbol2]), label] = np.nan
    return df

def filter_df_by_datetime(df, start_date=None, end_date=None, start_hour="00:00", end_hour="23:59"):
    df = df.copy()
    df['hour'] = pd.to_datetime(df['hour'])
    if start_date:
        df = df[df['hour'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['hour'] <= pd.to_datetime(end_date)]
    df['time_only'] = df['hour'].dt.time
    start_t = dt.strptime(start_hour, "%H:%M").time()
    end_t = dt.strptime(end_hour, "%H:%M").time()
    df = df[(df['time_only'] >= start_t) & (df['time_only'] <= end_t)]
    df.drop(columns=['time_only'], inplace=True)
    return df

def write_to_postgres(df, table_name, engine):
    if df.empty:
        logging.warning("DataFrame is empty. Nothing was written to the database.")
        return
    try:
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        logging.info(f"✅ Data written to table '{table_name}' in PostgreSQL.")
    except Exception as e:
        logging.error(f"❌ Failed to write to database: {e}")

# ---------- MAIN LOGIC ----------

def main(show_gui=False):
    logging.info("🚀 Starting pipeline...")

    df = pd.read_csv(FILE_PATH)
    df = compute_log_returns(df)
    df = compute_period_std(df)
    df = compute_rolling_std(df, ROLLING_WINDOWS)

    # Correlations
    correlation_pairs = [('SOL', 'WBTC'), ('SOL', 'WETH'), ('WBTC', 'WETH')]
    for (a, b) in correlation_pairs:
        for days, hours in {'30d': 720, '60d': 1440, '90d': 2160}.items():
            label = f"{a}{b}_{days}_Corr"
            df = compute_rolling_correlation(df, a, b, hours, label)

    # Filter time window
    df = filter_df_by_datetime(df, start_date="2023-01-01", end_date="2025-08-01", start_hour="00:00", end_hour="23:00")

    # Connect to PostgreSQL and write
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    write_to_postgres(df, TABLE_NAME, engine)

    logging.info("🎯 Done.")

# ---------- ENTRY POINT ----------

if __name__ == "__main__":
    main(show_gui=False)  # Set to True if you want to open the table viewer
