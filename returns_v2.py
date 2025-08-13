import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# === DB CONNECTION (edit these) ===
DB_NAME = 'crypto_data'
DB_USER = 'postgres'
DB_PASSWORD = 'yourpassword'   # <-- change
DB_HOST = 'localhost'
DB_PORT = 5432

# === QUERY PARAMS ===
# Set SYMBOLS=None to include ALL symbols present in the date range
SYMBOLS = None # e.g., ['SOL'] to force a subset
START   = '2025-01-01'
END     = '2025-07-31'

# === WINDOW PARAM (hours) for RA, plots, and VaR/ES ===
# 1=hourly, 6=6h, 12=12h, 24=daily, etc.
WINDOW_HOURS = 

# === CONNECT & LOAD FROM POSTGRES ===
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

if SYMBOLS:
    symbols_list = "', '".join(SYMBOLS)
    query = f"""
    SELECT *
    FROM crypto_metrics
    WHERE symbol IN ('{symbols_list}')
      AND hour >= '{START}'
      AND hour <= '{END}';
    """
else:
    query = f"""
    SELECT *
    FROM crypto_metrics
    WHERE hour >= '{START}'
      AND hour <= '{END}';
    """

df = pd.read_sql(query, engine)

# Ensure datetime & sort
df['hour'] = pd.to_datetime(df['hour'])
df = df.sort_values(['symbol', 'hour'])

# Derive symbols if not provided
if not SYMBOLS:
    SYMBOLS = sorted(df['symbol'].dropna().unique().tolist())

# -------------------------------------------------------------------
# Part 1: Descriptive Analysis of Returns (RA) on WINDOW_HOURS buckets
# -------------------------------------------------------------------
df['bucket'] = df['hour'].dt.floor(f'{WINDOW_HOURS}h')

agg = (
    df.groupby(['symbol', 'bucket'], as_index=False)
      .agg(hour=('hour', 'last'), close=('close', 'last'))
      .sort_values(['symbol', 'hour'])
)
agg['simple_return'] = agg.groupby('symbol')['close'].pct_change()

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

    def max_sequence(cond_series):
        m = c = 0
        for v in cond_series:
            c = c + 1 if v else 0
            m = max(m, c)
        return m

    max_up_seq = max_sequence(up_mask.fillna(False))
    max_down_seq = max_sequence(down_mask.fillna(False))
    ra.loc['Max Sequence'] = [max_up_seq, max_down_seq, max(max_up_seq, max_down_seq)]
    ra = ra.round(2)

    ups = sym_df[up_mask].nlargest(3, 'simple_return')
    downs = sym_df[down_mask].nsmallest(3, 'simple_return')
    best = pd.DataFrame({
        'Return (%)': list(ups['simple_return'] * 100) + list(downs['simple_return'] * 100),
        'DateTime':   list(ups['hour'])                 + list(downs['hour'])
    }, index=['1 Best', '2 Best', '3 Best', '1 Worst', '2 Worst', '3 Worst'])
    best['Return (%)'] = best['Return (%)'].round(2)
    return ra, best

label = f"{WINDOW_HOURS}-hour"
for sym in SYMBOLS:
    sym_df = agg[agg['symbol'] == sym].reset_index(drop=True)
    RA, Best = summarize_symbol(sym_df)

    print(f"\n=== {sym} | Descriptive Analysis of Returns ({label} period) ===")
    print(RA)
    print("\n-- Best & Worst Moves --")
    print(Best)
    print("\n")

# -------------------------------------------------------------------
# Part 1.5: Plot % returns bars per symbol (green up, red down)
# -------------------------------------------------------------------
plt.style.use('dark_background')  # comment out for light theme

def plot_return_bars(sym_df: pd.DataFrame, symbol: str, window_hours: int):
    x = sym_df['hour']
    y_pct = sym_df['simple_return'] * 100
    pos = y_pct.where(y_pct > 0)
    neg = y_pct.where(y_pct < 0)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.vlines(x[pos.notna()], 0, pos.dropna(), linewidth=0.5, color='limegreen', label=f'{symbol} Positive')
    ax.vlines(x[neg.notna()], 0, neg.dropna(), linewidth=0.5, color='red', label=f'{symbol} Negative')

    ax.axhline(0, linewidth=0.8, color='gray', alpha=0.7)
    ax.set_title(f"{symbol}: {window_hours}h % Returns", fontsize=14, weight='bold')
    ax.set_ylabel("% Returns")
    ax.set_xlabel("Date")
    ax.legend(handles=[Patch(color='limegreen', label=f'{symbol} Positive'),
                       Patch(color='red', label=f'{symbol} Negative')],
              loc='upper right')
    ax.margins(x=0)
    fig.tight_layout()

    # Save per symbol (comment out if you don't want files)
    out_path = f"{symbol}_{window_hours}h_returns.png"
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

for sym in SYMBOLS:
    sym_df = agg[agg['symbol'] == sym].dropna(subset=['simple_return']).reset_index(drop=True)
    if not sym_df.empty:
        plot_return_bars(sym_df, sym, WINDOW_HOURS)

# -------------------------------------------------------------------
# Part 2: VaR & ES on WINDOW_HOURS returns (pivoted only)
# -------------------------------------------------------------------
risk = (
    df.assign(period=df['hour'].dt.floor(f'{WINDOW_HOURS}h'))
      .sort_values(['symbol', 'hour'])
      .groupby(['symbol', 'period'], as_index=False)
      .agg(close=('close', 'last'))
      .sort_values(['symbol', 'period'])
)
risk['ret'] = risk.groupby('symbol')['close'].pct_change()

# Tail probabilities & labels (reported as negative % losses)
tail_probs  = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05]  # 0.10%, 0.30%, 0.50%, 1%, 2%, 3%, 5%
tail_labels = ['0.10', '0.30', '0.50', '1.00', '2.00', '3.00', '5.00']

def expected_shortfall(series: pd.Series, alpha: float) -> float:
    """Left-tail Expected Shortfall at level alpha (returns decimal)."""
    if series.empty:
        return float('nan')
    q = series.quantile(alpha, interpolation='linear')
    tail = series[series <= q]
    if tail.empty:
        return float('nan')
    return tail.mean()

var_rows, es_rows = [], []
for sym in SYMBOLS:
    r = risk.loc[risk['symbol'] == sym, 'ret'].dropna()
    if r.empty:
        var_rows.append([float('nan')] * len(tail_probs))
        es_rows.append([float('nan')] * len(tail_probs))
        continue
    var_rows.append([(r.quantile(p, interpolation='linear') * 100) for p in tail_probs])
    es_rows.append([(expected_shortfall(r, p) * 100) for p in tail_probs])

var_pivot = pd.DataFrame(var_rows, index=SYMBOLS, columns=tail_labels).T
es_pivot  = pd.DataFrame(es_rows,  index=SYMBOLS, columns=tail_labels).T

# Keep desired order (already ordered, but enforce just in case)
order_map = {lab: i for i, lab in enumerate(tail_labels)}
var_pivot = var_pivot.sort_index(key=lambda idx: idx.map(order_map))
es_pivot  = es_pivot.sort_index(key=lambda idx: idx.map(order_map))

var_pivot.index.name = "Tail Probability (%)"
es_pivot.index.name  = "Tail Probability (%)"
var_pivot.columns.name = "Symbol"
es_pivot.columns.name  = "Symbol"

print(f"=== {label} Value-at-Risk (negative % loss) ===")
print(var_pivot.round(2))
print(f"\n=== {label} Expected Shortfall (negative % loss) ===")
print(es_pivot.round(2))

# -------------------------------------------------------------------
# Part 2.5: Histogram of WINDOW_HOURS returns + VaR lines per symbol
# -------------------------------------------------------------------
def plot_return_hist(sym_ret_pct: pd.Series, symbol: str, window_hours: int):
    """Histogram of % returns with VaR lines and clean labels."""
    y = sym_ret_pct.dropna()
    if y.empty:
        return

    # VaR (in %)
    var_vals = [y.quantile(p, interpolation='linear') for p in tail_probs]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.hist(y, bins=80, alpha=0.85, color='steelblue', edgecolor='black')

    colors = ['darkred', 'firebrick', 'indianred', 'orange', 'gold', 'yellowgreen', 'green']

    # Vertical dashed lines + labels at top
    ymax = ax.get_ylim()[1]
    for p, v, c in zip(tail_probs, var_vals, colors):
        ax.axvline(v, linestyle='--', linewidth=1.5, color=c)
        ax.text(
            v, ymax * 0.95,  # position near top
            f"VaR {p*100:.2f}%\n{v:.2f}%",  # show percentile + value
            ha='center', va='top',
            fontsize=9, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.8),
            color='black'
        )

    ax.set_title(f"Distribution of {window_hours}h Returns for {symbol}", fontsize=14, weight='bold')
    ax.set_xlabel(f"{window_hours}h Return %")
    ax.set_ylabel("Frequency")

    # Zoom in on central range
    x_lo = y.quantile(0.001)
    x_hi = y.quantile(0.999)
    ax.set_xlim(x_lo, x_hi)

    fig.tight_layout()
    out_path = f"{symbol}_{window_hours}h_returns_hist.png"
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


for sym in SYMBOLS:
    sym_pct = risk.loc[risk['symbol'] == sym, 'ret'].dropna() * 100  # convert to %
    if not sym_pct.empty:
        plot_return_hist(sym_pct, sym, WINDOW_HOURS)
