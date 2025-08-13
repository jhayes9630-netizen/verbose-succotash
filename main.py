# main.py
from typing import List, Optional, Dict, Any
import os

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine

# ----------------------------
# Config from environment vars
# ----------------------------
DB_NAME = os.getenv("DB_NAME", "crypto_data")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "yourpassword")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

def get_engine():
    dsn = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(dsn)

# ----------------------------
# FastAPI app + CORS
# ----------------------------
app = FastAPI(title="Crypto Return Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Schemas
# ----------------------------
class AnalyzeRequest(BaseModel):
    window_hours: int = Field(6, gt=0, description="Bucket size in hours (1=hourly, 24=daily, etc.)")
    start: str = "2025-01-01"  # YYYY-MM-DD
    end: str   = "2025-07-31"
    symbols: Optional[List[str]] = None  # None or [] => all symbols in range

class TableData(BaseModel):
    columns: List[str]
    index: List[str]
    data: List[List[Any]]

class SymbolOutputs(BaseModel):
    ra: TableData
    best: TableData
    # simple chart series (dates ISO strings, values in %)
    bar_series: Dict[str, List]      # {x: [], y: []}
    hist: Dict[str, List]            # {bins: [], counts: []}
    var_lines: Dict[str, List]       # {levels: [], values: []} (% values)

class AnalyzeResponse(BaseModel):
    symbols: List[str]
    var_pivot: TableData
    es_pivot: TableData
    per_symbol: Dict[str, SymbolOutputs]

# ----------------------------
# Helpers
# ----------------------------
def max_sequence(cond_series: pd.Series) -> int:
    m = c = 0
    for v in cond_series:
        c = c + 1 if bool(v) else 0
        m = max(m, c)
    return m

def expected_shortfall(s: pd.Series, alpha: float) -> float:
    if s.empty:
        return float('nan')
    q = s.quantile(alpha, interpolation='linear')
    tail = s[s <= q]
    return float('nan') if tail.empty else float(tail.mean())

def summarize_symbol(sym_df: pd.DataFrame):
    ra = pd.DataFrame(index=['Number', 'Percentage', 'Average', 'Standard Dev.', 'Max Sequence'],
                      columns=['up', 'down', 'total'], dtype='float64')

    up_mask = sym_df['simple_return'] > 0
    down_mask = sym_df['simple_return'] < 0
    total = sym_df['simple_return'].notna().sum()

    ra.loc['Number'] = [up_mask.sum(), down_mask.sum(), total]
    ra.loc['Percentage'] = (ra.loc['Number'] / total) * 100 if total else [0, 0, 0]
    ra.loc['Average'] = [
        sym_df.loc[up_mask, 'simple_return'].mean() * 100 if up_mask.any() else 0.0,
        sym_df.loc[down_mask, 'simple_return'].mean() * 100 if down_mask.any() else 0.0,
        sym_df['simple_return'].mean() * 100 if total else 0.0
    ]
    ra.loc['Standard Dev.'] = [
        sym_df.loc[up_mask, 'simple_return'].std() * 100 if up_mask.sum() > 1 else 0.0,
        sym_df.loc[down_mask, 'simple_return'].std() * 100 if down_mask.sum() > 1 else 0.0,
        sym_df['simple_return'].std() * 100 if total > 1 else 0.0
    ]

    def _max_seq(s: pd.Series) -> int:
        m = c = 0
        for v in s:
            c = c + 1 if bool(v) else 0
            m = max(m, c)
        return m

    max_up_seq = _max_seq(up_mask.fillna(False))
    max_down_seq = _max_seq(down_mask.fillna(False))
    ra.loc['Max Sequence'] = [max_up_seq, max_down_seq, max(max_up_seq, max_down_seq)]
    ra = ra.round(2)

    # Best/Worst moves (top 3 each)
    ups = sym_df[up_mask].nlargest(3, 'simple_return')
    downs = sym_df[down_mask].nsmallest(3, 'simple_return')
    best = pd.DataFrame({
        'Return (%)': list(ups['simple_return'] * 100) + list(downs['simple_return'] * 100),
        'DateTime':   list(ups['hour'])                 + list(downs['hour'])
    }, index=['1 Best', '2 Best', '3 Best', '1 Worst', '2 Worst', '3 Worst'])
    best['Return (%)'] = best['Return (%)'].round(2)
    return ra, best

# ----------------------------
# Basic routes so root isn't 404
# ----------------------------
@app.get("/")
def root():
    return {
        "message": "Welcome to the Crypto Analysis API",
        "usage": "POST JSON to /analyze or open /docs for interactive testing"
    }

@app.get("/ping")
def ping():
    return {"pong": True}

# ----------------------------
# Main endpoint
# ----------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    engine = get_engine()

    # Build query
    if req.symbols:
        syms = "', '".join(req.symbols)
        query = f"""
        SELECT * FROM crypto_metrics
        WHERE symbol IN ('{syms}')
          AND hour >= '{req.start}' AND hour <= '{req.end}';
        """
    else:
        query = f"""
        SELECT * FROM crypto_metrics
        WHERE hour >= '{req.start}' AND hour <= '{req.end}';
        """

    df = pd.read_sql(query, engine)
    if df.empty:
        # empty but valid response structure
        empty_td = TableData(columns=[], index=[], data=[])
        return AnalyzeResponse(symbols=[], var_pivot=empty_td, es_pivot=empty_td, per_symbol={})

    # Prep
    df['hour'] = pd.to_datetime(df['hour'])
    df = df.sort_values(['symbol', 'hour'])
    symbols = req.symbols or sorted(df['symbol'].dropna().unique().tolist())

    # Aggregate to window buckets (lowercase 'h' to avoid pandas warning)
    df['bucket'] = df['hour'].dt.floor(f'{req.window_hours}h')
    agg = (
        df.groupby(['symbol', 'bucket'], as_index=False)
          .agg(hour=('hour', 'last'), close=('close', 'last'))
          .sort_values(['symbol', 'hour'])
    )
    agg['simple_return'] = agg.groupby('symbol')['close'].pct_change()

    # Same horizon series for VaR/ES + histogram
    risk = (
        df.assign(period=df['hour'].dt.floor(f'{req.window_hours}h'))
          .sort_values(['symbol', 'hour'])
          .groupby(['symbol', 'period'], as_index=False)
          .agg(close=('close', 'last'))
          .sort_values(['symbol', 'period'])
    )
    risk['ret'] = risk.groupby('symbol')['close'].pct_change()

    # Tail probabilities/labels
    tail_probs  = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05]  # 0.10%, ..., 5%
    tail_labels = ['0.10', '0.30', '0.50', '1.00', '2.00', '3.00', '5.00']

    # VaR / ES pivot tables
    var_rows, es_rows = [], []
    for sym in symbols:
        r = risk.loc[risk['symbol'] == sym, 'ret'].dropna()
        if r.empty:
            var_rows.append([float('nan')] * len(tail_probs))
            es_rows.append([float('nan')] * len(tail_probs))
            continue
        var_rows.append([(r.quantile(p, interpolation='linear') * 100) for p in tail_probs])
        es_rows.append([(expected_shortfall(r, p) * 100) for p in tail_probs])

    var_pivot = pd.DataFrame(var_rows, index=symbols, columns=tail_labels).T.round(2)
    es_pivot  = pd.DataFrame(es_rows,  index=symbols, columns=tail_labels).T.round(2)

    # Per-symbol RA/Best + chart series
    per_symbol: Dict[str, SymbolOutputs] = {}
    for sym in symbols:
        sym_df = agg[agg['symbol'] == sym].reset_index(drop=True)
        RA, Best = summarize_symbol(sym_df)

        # Bar series (%)
        bar_x = sym_df['hour'].astype(str).tolist()
        bar_y = (sym_df['simple_return'] * 100).fillna(0).tolist()

        # Histogram + VaR lines (%)
        sym_ret_pct = (risk.loc[risk['symbol'] == sym, 'ret'] * 100).dropna()
        if sym_ret_pct.empty:
            bins = []; counts = []
            var_vals = [float('nan')] * len(tail_probs)
        else:
            import numpy as np
            counts_np, bin_edges = np.histogram(sym_ret_pct, bins=80)
            bins = bin_edges.tolist()
            counts = counts_np.tolist()
            var_vals = [float(sym_ret_pct.quantile(p, interpolation='linear')) for p in tail_probs]

        per_symbol[sym] = SymbolOutputs(
            ra=TableData(columns=list(RA.columns), index=list(RA.index), data=RA.values.tolist()),
            best=TableData(columns=list(Best.columns), index=list(Best.index), data=Best.values.tolist()),
            bar_series={"x": bar_x, "y": bar_y},
            hist={"bins": bins, "counts": counts},
            var_lines={"levels": tail_labels, "values": var_vals}
        )

    return AnalyzeResponse(
        symbols=symbols,
        var_pivot=TableData(columns=list(var_pivot.columns), index=list(var_pivot.index), data=var_pivot.values.tolist()),
        es_pivot=TableData(columns=list(es_pivot.columns), index=list(es_pivot.index), data=es_pivot.values.tolist()),
        per_symbol=per_symbol
    )
