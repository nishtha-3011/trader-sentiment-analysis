import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ── Styling 
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'text.color': '#e6edf3',
    'grid.color': '#21262d',
    'grid.alpha': 0.5,
    'figure.titlesize': 16,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'legend.labelcolor': '#e6edf3',
})

FEAR_COLOR = '#ff7b72'
GREED_COLOR = '#3fb950'
NEUTRAL_COLOR = '#58a6ff'
ACCENT = '#d2a8ff'

CHARTS = '/home/claude/primetrade_analysis/charts'
OUTPUTS = '/home/claude/primetrade_analysis/outputs'

print("=" * 60)
print("PART A — DATA PREPARATION")
print("=" * 60)

# ── Load Datasets 
fg_raw = pd.read_csv('/mnt/user-data/uploads/1776341468992_fear_greed_index.csv')
hd_raw = pd.read_csv('/mnt/user-data/uploads/1776341479837_historical_data.csv')

print(f"\n[Fear/Greed Index]  Rows: {fg_raw.shape[0]}, Cols: {fg_raw.shape[1]}")
print(f"[Historical Trades] Rows: {hd_raw.shape[0]}, Cols: {hd_raw.shape[1]}")
print("\nFear/Greed columns:", fg_raw.columns.tolist())
print("Trades columns:", hd_raw.columns.tolist())

# ── Missing Values 
print("\n--- Missing Values ---")
print("Fear/Greed:", fg_raw.isnull().sum().to_dict())
print("Trades:", hd_raw.isnull().sum().to_dict())

# ── Duplicates 
fg_dups = fg_raw.duplicated().sum()
hd_dups = hd_raw.duplicated().sum()
print(f"\nFear/Greed duplicates: {fg_dups}")
print(f"Trades duplicates: {hd_dups}")

# ── Date Alignment 
fg = fg_raw.copy()
fg['date'] = pd.to_datetime(fg['date']).dt.normalize()

# Simplify sentiment: Fear/Extreme Fear → Fear, Greed/Extreme Greed → Greed, Neutral → Neutral
def simplify_sentiment(c):
    c = c.lower()
    if 'fear' in c:
        return 'Fear'
    elif 'greed' in c:
        return 'Greed'
    else:
        return 'Neutral'

fg['sentiment'] = fg['classification'].apply(simplify_sentiment)
print("\nSentiment distribution:")
print(fg['sentiment'].value_counts())

hd = hd_raw.copy()
hd['datetime'] = pd.to_datetime(hd['Timestamp IST'], format='%d-%m-%Y %H:%M')
hd['date'] = hd['datetime'].dt.normalize()
hd.rename(columns={
    'Account': 'account',
    'Coin': 'coin',
    'Execution Price': 'exec_price',
    'Size Tokens': 'size_tokens',
    'Size USD': 'size_usd',
    'Side': 'side',
    'Direction': 'direction',
    'Closed PnL': 'closed_pnl',
    'Fee': 'fee',
    'Start Position': 'start_position',
    'Crossed': 'crossed',
}, inplace=True)

# Derive leverage proxy (Size USD / abs(start_position) when start_position != 0)
hd['leverage_proxy'] = np.where(
    hd['start_position'].abs() > 0,
    (hd['size_usd'] / hd['start_position'].abs()).clip(1, 100),
    np.nan
)

# Net PnL after fee
hd['net_pnl'] = hd['closed_pnl'] - hd['fee']

# Merge sentiment onto trades
df = hd.merge(fg[['date', 'sentiment', 'value']], on='date', how='inner')
print(f"\nMerged dataset rows: {len(df)}")
print(f"Date range in merged: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Unique accounts: {df['account'].nunique()}")

# ── Key Metrics per Day per Account 
close_mask = df['direction'].str.contains('Close|Sell|Settlement|Liquidat', case=False, na=False)
closed_trades = df[close_mask].copy()

daily_account = closed_trades.groupby(['date', 'account', 'sentiment']).agg(
    daily_pnl=('net_pnl', 'sum'),
    num_trades=('net_pnl', 'count'),
    win_trades=('net_pnl', lambda x: (x > 0).sum()),
    avg_size_usd=('size_usd', 'mean'),
    avg_leverage=('leverage_proxy', 'mean'),
).reset_index()

daily_account['win_rate'] = daily_account['win_trades'] / daily_account['num_trades']

# Long/short ratio per day per account
ls = df[df['direction'].isin(['Open Long','Open Short'])].groupby(
    ['date','account','sentiment']
).apply(
    lambda x: (x['direction']=='Open Long').sum() / max((x['direction']=='Open Short').sum(), 1)
).reset_index(name='long_short_ratio')

daily_account = daily_account.merge(ls, on=['date','account','sentiment'], how='left')

# Save daily metrics table
daily_account.to_csv(f'{OUTPUTS}/daily_account_metrics.csv', index=False)
print(f"\nDaily metrics table saved: {daily_account.shape}")

print("\n" + "=" * 60)
print("PART B — ANALYSIS")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Q1: Performance differences — Fear vs Greed days
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q1] Performance: Fear vs Greed Days")

perf = daily_account[daily_account['sentiment'].isin(['Fear','Greed'])]
stats = perf.groupby('sentiment').agg(
    mean_pnl=('daily_pnl','mean'),
    median_pnl=('daily_pnl','median'),
    std_pnl=('daily_pnl','std'),
    mean_winrate=('win_rate','mean'),
    mean_leverage=('avg_leverage','mean'),
).round(3)
print(stats)
stats.to_csv(f'{OUTPUTS}/fear_greed_performance.csv')

# Drawdown proxy: worst daily PnL per account per sentiment period
drawdown = perf.groupby(['account','sentiment'])['daily_pnl'].min().reset_index()
drawdown.columns = ['account','sentiment','max_drawdown']
drawdown_summary = drawdown.groupby('sentiment')['max_drawdown'].mean()
print("\nAvg Max Drawdown per Sentiment:")
print(drawdown_summary)

# ── CHART 1: PnL Distribution Fear vs Greed 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Chart 1 — Trader Performance: Fear vs Greed Days', fontweight='bold', y=1.01)

# 1a: Daily PnL Distribution
ax = axes[0]
for sent, color in [('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]:
    data = perf[perf['sentiment'] == sent]['daily_pnl']
    data_clipped = data.clip(-5000, 5000)
    ax.hist(data_clipped, bins=60, alpha=0.65, color=color, label=sent, density=True)
    ax.axvline(data.median(), color=color, linestyle='--', linewidth=1.5, alpha=0.9)
ax.set_title('Daily PnL Distribution (clipped ±5K)')
ax.set_xlabel('Daily Net PnL (USD)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)

# 1b: Win Rate Box
ax = axes[1]
fear_wr = perf[perf['sentiment']=='Fear']['win_rate'].dropna()
greed_wr = perf[perf['sentiment']=='Greed']['win_rate'].dropna()
bp = ax.boxplot([fear_wr, greed_wr], patch_artist=True,
                medianprops=dict(color='white', linewidth=2))
bp['boxes'][0].set_facecolor(FEAR_COLOR); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(GREED_COLOR); bp['boxes'][1].set_alpha(0.7)
ax.set_xticklabels(['Fear', 'Greed'])
ax.set_title('Win Rate Distribution')
ax.set_ylabel('Win Rate')
ax.grid(True, alpha=0.3)

# 1c: Mean PnL bar
ax = axes[2]
sentiments = ['Fear','Greed']
means = [perf[perf['sentiment']==s]['daily_pnl'].mean() for s in sentiments]
colors = [FEAR_COLOR, GREED_COLOR]
bars = ax.bar(sentiments, means, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'${val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Mean Daily PnL per Sentiment')
ax.set_ylabel('Mean Net PnL (USD)')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='white', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(f'{CHARTS}/chart1_performance_fear_greed.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Chart 1 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# Q2: Behavioral Changes based on Sentiment
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q2] Behavioral Changes")

behavior = daily_account[daily_account['sentiment'].isin(['Fear','Greed'])].groupby('sentiment').agg(
    avg_trades_per_day=('num_trades','mean'),
    avg_leverage=('avg_leverage','mean'),
    avg_size_usd=('avg_size_usd','mean'),
    avg_ls_ratio=('long_short_ratio','mean'),
).round(3)
print(behavior)
behavior.to_csv(f'{OUTPUTS}/behavior_by_sentiment.csv')

# ── CHART 2: Behavior — Trade Frequency, Leverage, Long/Short 
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Chart 2 — Trader Behavior: Fear vs Greed Days', fontweight='bold', y=1.01)

metrics = [
    ('num_trades', 'Avg Trades/Day', '# Trades'),
    ('avg_leverage', 'Avg Leverage Proxy', 'Leverage (×)'),
    ('avg_size_usd', 'Avg Trade Size (USD)', 'USD'),
    ('long_short_ratio', 'Long/Short Ratio', 'Ratio'),
]
for ax, (col, title, ylabel) in zip(axes, metrics):
    for sent, color in [('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]:
        data = perf[perf['sentiment'] == sent][col].dropna()
        ax.hist(data.clip(data.quantile(0.01), data.quantile(0.99)),
                bins=40, alpha=0.6, color=color, label=sent, density=True)
        ax.axvline(data.median(), color=color, linestyle='--', linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel('Density')
    ax.set_xlabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CHARTS}/chart2_behavior_fear_greed.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Chart 2 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# Q3: Trader Segmentation — 3 Segments
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q3] Trader Segmentation")

# Aggregate per-account stats
account_stats = closed_trades.groupby('account').agg(
    total_pnl=('net_pnl','sum'),
    total_trades=('net_pnl','count'),
    win_trades=('net_pnl', lambda x: (x>0).sum()),
    avg_leverage=('leverage_proxy','mean'),
    avg_size_usd=('size_usd','mean'),
    std_pnl=('net_pnl','std'),
).reset_index()
account_stats['win_rate'] = account_stats['win_trades'] / account_stats['total_trades']
account_stats['avg_daily_trades'] = account_stats['total_trades'] / closed_trades['date'].nunique()

# SEGMENT 1: High vs Low Leverage
lev_q = account_stats['avg_leverage'].quantile([0.33, 0.67])
account_stats['leverage_seg'] = pd.cut(
    account_stats['avg_leverage'],
    bins=[-np.inf, lev_q[0.33], lev_q[0.67], np.inf],
    labels=['Low Leverage', 'Medium Leverage', 'High Leverage']
)

# SEGMENT 2: Frequent vs Infrequent Traders
freq_median = account_stats['avg_daily_trades'].median()
account_stats['freq_seg'] = np.where(
    account_stats['avg_daily_trades'] >= freq_median, 'Frequent', 'Infrequent'
)

# SEGMENT 3: Consistent Winners vs Inconsistent (by win rate & PnL stability)
pnl_q50 = account_stats['total_pnl'].median()
wr_q50 = account_stats['win_rate'].median()
def classify_consistency(row):
    if row['total_pnl'] > pnl_q50 and row['win_rate'] > wr_q50:
        return 'Consistent Winner'
    elif row['total_pnl'] < 0:
        return 'Consistent Loser'
    else:
        return 'Inconsistent'
account_stats['consistency_seg'] = account_stats.apply(classify_consistency, axis=1)

print("\nLeverage Segments:")
print(account_stats.groupby('leverage_seg')[['total_pnl','win_rate','avg_daily_trades']].mean().round(2))
print("\nFrequency Segments:")
print(account_stats.groupby('freq_seg')[['total_pnl','win_rate','avg_leverage']].mean().round(2))
print("\nConsistency Segments:")
print(account_stats.groupby('consistency_seg')[['total_pnl','win_rate','avg_daily_trades']].mean().round(2))

account_stats.to_csv(f'{OUTPUTS}/account_segments.csv', index=False)

# ── CHART 3: Segmentation Plots 
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Chart 3 — Trader Segmentation Analysis', fontweight='bold', y=1.01)

# 3a: Leverage seg PnL
ax = axes[0]
seg_order = ['Low Leverage','Medium Leverage','High Leverage']
colors_lev = ['#3fb950','#58a6ff','#ff7b72']
means_lev = [account_stats[account_stats['leverage_seg']==s]['total_pnl'].mean() for s in seg_order]
bars = ax.bar(seg_order, means_lev, color=colors_lev, alpha=0.85, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, means_lev):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'${val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Avg Total PnL by Leverage Segment')
ax.set_ylabel('Total PnL (USD)')
ax.set_xticklabels(seg_order, rotation=12, fontsize=9)
ax.axhline(0, color='white', linewidth=0.5, alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# 3b: Frequency seg — win rate and trade count
ax = axes[1]
x = np.arange(2)
freq_segs = ['Frequent','Infrequent']
wr_vals = [account_stats[account_stats['freq_seg']==s]['win_rate'].mean() for s in freq_segs]
lev_vals = [account_stats[account_stats['freq_seg']==s]['avg_leverage'].mean() for s in freq_segs]
bars1 = ax.bar(x - 0.2, wr_vals, width=0.35, color=GREED_COLOR, alpha=0.8, label='Win Rate')
ax2b = ax.twinx()
ax2b.set_ylabel('Avg Leverage', color=ACCENT)
bars2 = ax2b.bar(x + 0.2, lev_vals, width=0.35, color=ACCENT, alpha=0.8, label='Avg Leverage')
ax2b.tick_params(axis='y', labelcolor=ACCENT)
ax.set_xticks(x); ax.set_xticklabels(freq_segs)
ax.set_title('Frequent vs Infrequent Traders')
ax.set_ylabel('Win Rate')
ax.grid(True, alpha=0.3, axis='y')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

# 3c: Consistency pie
ax = axes[2]
cons_counts = account_stats['consistency_seg'].value_counts()
colors_cons = {'Consistent Winner': GREED_COLOR, 'Inconsistent': '#f0a500', 'Consistent Loser': FEAR_COLOR}
pie_colors = [colors_cons.get(k, '#888') for k in cons_counts.index]
wedges, texts, autotexts = ax.pie(
    cons_counts.values, labels=cons_counts.index, colors=pie_colors,
    autopct='%1.1f%%', startangle=90,
    textprops={'color':'#e6edf3', 'fontsize': 9},
    wedgeprops={'edgecolor':'#0d1117', 'linewidth': 2}
)
for at in autotexts:
    at.set_fontsize(9); at.set_fontweight('bold')
ax.set_title('Trader Consistency Distribution')
ax.set_facecolor('#161b22')

plt.tight_layout()
plt.savefig(f'{CHARTS}/chart3_segmentation.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Chart 3 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# Q4: 3+ Insights — backed by charts
# ─────────────────────────────────────────────────────────────────────────────

# ── CHART 4: Insight — Sentiment over Time + PnL heatmap 
daily_pnl_sentiment = df[close_mask].groupby(['date','sentiment']).agg(
    total_pnl=('net_pnl','sum'),
    num_trades=('net_pnl','count'),
).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(16, 9))
fig.suptitle('Chart 4 — Insight: Market Sentiment vs Daily PnL Over Time', fontweight='bold')

ax = axes[0]
sent_colors = {'Fear': FEAR_COLOR, 'Greed': GREED_COLOR, 'Neutral': NEUTRAL_COLOR}
for sent in ['Fear','Greed','Neutral']:
    sub = daily_pnl_sentiment[daily_pnl_sentiment['sentiment']==sent]
    ax.scatter(sub['date'], sub['total_pnl'], color=sent_colors[sent],
               alpha=0.55, s=15, label=sent)
# Rolling 30d
all_sorted = daily_pnl_sentiment.groupby('date')['total_pnl'].sum().reset_index().sort_values('date')
roll = all_sorted.set_index('date')['total_pnl'].rolling('30D').mean()
ax.plot(roll.index, roll.values, color='white', linewidth=1.5, alpha=0.7, label='30D Rolling Avg')
ax.set_ylabel('Total Daily PnL (USD)')
ax.set_title('Daily Total PnL by Sentiment Period')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)

ax = axes[1]
# Win rate by month and sentiment
daily_pnl_sentiment['month'] = pd.to_datetime(daily_pnl_sentiment['date']).dt.to_period('M').astype(str)
monthly = df[close_mask].copy()
monthly['month'] = monthly['date'].dt.to_period('M').astype(str)
monthly_wr = monthly.groupby(['month','sentiment']).apply(
    lambda x: (x['net_pnl']>0).sum()/len(x)
).reset_index(name='win_rate')
monthly_pivot = monthly_wr.pivot(index='month', columns='sentiment', values='win_rate').fillna(0)

months = monthly_pivot.index.tolist()
x = np.arange(len(months))
width = 0.25
for i, (col, color) in enumerate([('Fear',FEAR_COLOR),('Greed',GREED_COLOR),('Neutral',NEUTRAL_COLOR)]):
    if col in monthly_pivot.columns:
        ax.bar(x + i*width, monthly_pivot[col], width=width, color=color, alpha=0.8, label=col)
ax.set_xticks(x + width); ax.set_xticklabels(months, rotation=45, ha='right', fontsize=7)
ax.set_title('Monthly Win Rate by Sentiment Type')
ax.set_ylabel('Win Rate')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0.5, color='white', linewidth=0.8, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{CHARTS}/chart4_insight_time_pnl.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Chart 4 saved.")

# ── CHART 5: Insight — Leverage vs PnL scatter, colored by Sentiment 
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Chart 5 — Insight: Leverage & Position Size vs PnL by Sentiment', fontweight='bold')

ax = axes[0]
for sent, color, marker in [('Fear', FEAR_COLOR, 'o'), ('Greed', GREED_COLOR, 's')]:
    sub = daily_account[daily_account['sentiment']==sent].dropna(subset=['avg_leverage','daily_pnl'])
    sub_clipped = sub[sub['daily_pnl'].between(-5000,5000)]
    ax.scatter(sub_clipped['avg_leverage'].clip(0,50), sub_clipped['daily_pnl'],
               color=color, alpha=0.3, s=20, label=sent, marker=marker)
ax.axhline(0, color='white', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Avg Leverage Proxy')
ax.set_ylabel('Daily PnL (USD)')
ax.set_title('Leverage vs Daily PnL')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
for sent, color in [('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]:
    sub = daily_account[daily_account['sentiment']==sent].dropna(subset=['avg_size_usd','daily_pnl'])
    sub_clipped = sub[sub['daily_pnl'].between(-5000,5000) & (sub['avg_size_usd'] < 1e6)]
    ax.scatter(sub_clipped['avg_size_usd'], sub_clipped['daily_pnl'],
               color=color, alpha=0.3, s=20, label=sent)
ax.axhline(0, color='white', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Avg Position Size (USD)')
ax.set_ylabel('Daily PnL (USD)')
ax.set_title('Position Size vs Daily PnL')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CHARTS}/chart5_leverage_pnl.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Chart 5 saved.")

# ── CHART 6: Long/Short Ratio Heatmap by Sentiment + Account 
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Chart 6 — Insight: Trading Bias & Volume by Sentiment', fontweight='bold')

ax = axes[0]
ls_by_sent = daily_account[daily_account['sentiment'].isin(['Fear','Greed'])].groupby(
    ['account','sentiment']
)['long_short_ratio'].mean().reset_index()
ls_pivot = ls_by_sent.pivot(index='account', columns='sentiment', values='long_short_ratio').fillna(1)
ls_pivot.index = [f'Acc {i+1}' for i in range(len(ls_pivot))]
im = ax.imshow(ls_pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=3)
plt.colorbar(im, ax=ax, label='L/S Ratio (>1 = Long-biased)')
ax.set_xticks(range(len(ls_pivot.columns))); ax.set_xticklabels(ls_pivot.columns)
ax.set_yticks(range(len(ls_pivot.index))); ax.set_yticklabels(ls_pivot.index, fontsize=7)
ax.set_title('Long/Short Ratio Heatmap\n(per Trader per Sentiment)')
for i in range(len(ls_pivot)):
    for j in range(len(ls_pivot.columns)):
        ax.text(j, i, f'{ls_pivot.values[i,j]:.1f}',
                ha='center', va='center', color='black', fontsize=8, fontweight='bold')

ax = axes[1]
volume_by_sent = daily_account[daily_account['sentiment'].isin(['Fear','Greed'])].groupby('sentiment').agg(
    avg_trades=('num_trades','mean'),
    median_size=('avg_size_usd','median'),
).reset_index()
x_pos = np.arange(len(volume_by_sent))
bars1 = ax.bar(x_pos - 0.2, volume_by_sent['avg_trades'],
               width=0.35, color=[FEAR_COLOR, GREED_COLOR], alpha=0.8, label='Avg Trades/Day')
ax2 = ax.twinx()
bars2 = ax2.bar(x_pos + 0.2, volume_by_sent['median_size'],
                width=0.35, color=[FEAR_COLOR, GREED_COLOR], alpha=0.4, hatch='//')
ax.set_xticks(x_pos); ax.set_xticklabels(volume_by_sent['sentiment'])
ax.set_ylabel('Avg Trades / Day')
ax2.set_ylabel('Median Position Size (USD)')
ax.set_title('Trade Frequency & Size by Sentiment')
ax.grid(True, alpha=0.3, axis='y')
p1 = mpatches.Patch(color='grey', alpha=0.8, label='Avg Trades')
p2 = mpatches.Patch(color='grey', alpha=0.4, hatch='//', label='Median Size')
ax.legend(handles=[p1,p2], loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(f'{CHARTS}/chart6_trading_bias.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Chart 6 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# BONUS: K-Means Clustering
# ─────────────────────────────────────────────────────────────────────────────
print("\n[BONUS] K-Means Clustering of Traders")

cluster_features = account_stats[['total_pnl','avg_daily_trades','win_rate','avg_leverage','avg_size_usd']].copy()
cluster_features = cluster_features.fillna(cluster_features.median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
account_stats['cluster'] = kmeans.fit_predict(X_scaled)

cluster_summary = account_stats.groupby('cluster').agg(
    n_traders=('account','count'),
    avg_pnl=('total_pnl','mean'),
    avg_winrate=('win_rate','mean'),
    avg_leverage=('avg_leverage','mean'),
    avg_trades=('avg_daily_trades','mean'),
).round(2)
print(cluster_summary)
cluster_summary.to_csv(f'{OUTPUTS}/cluster_summary.csv')

# ── CHART 7: Cluster Visualization 
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Chart 7 — BONUS: Trader Behavioral Archetype Clusters', fontweight='bold')

cluster_names = {
    cluster_summary['avg_pnl'].idxmax(): '🏆 High Earners',
}
# simple label based on sorted pnl
sorted_clusters = cluster_summary['avg_pnl'].sort_values(ascending=False).index.tolist()
names_list = ['High Earners', 'Moderate Traders', 'Low Earners', 'Struggling Traders']
cluster_label_map = {c: n for c, n in zip(sorted_clusters, names_list)}
account_stats['cluster_name'] = account_stats['cluster'].map(cluster_label_map)

colors_c = ['#3fb950','#58a6ff','#f0a500','#ff7b72']
ax = axes[0]
for i, (clust, name) in enumerate(cluster_label_map.items()):
    mask = account_stats['cluster'] == clust
    ax.scatter(
        account_stats[mask]['avg_daily_trades'],
        account_stats[mask]['total_pnl'],
        color=colors_c[i], s=200, label=name, alpha=0.85, edgecolors='white', linewidth=0.8
    )
ax.set_xlabel('Avg Daily Trades')
ax.set_ylabel('Total PnL (USD)')
ax.set_title('Clusters: Trade Frequency vs Total PnL')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='white', linewidth=0.5, alpha=0.5)

ax = axes[1]
for i, (clust, name) in enumerate(cluster_label_map.items()):
    mask = account_stats['cluster'] == clust
    ax.scatter(
        account_stats[mask]['avg_leverage'].clip(0,50),
        account_stats[mask]['win_rate'],
        color=colors_c[i], s=200, label=name, alpha=0.85, edgecolors='white', linewidth=0.8
    )
ax.set_xlabel('Avg Leverage')
ax.set_ylabel('Win Rate')
ax.set_title('Clusters: Leverage vs Win Rate')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(0.5, color='white', linewidth=0.8, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{CHARTS}/chart7_clusters.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Chart 7 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# BONUS: Predictive Model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[BONUS] Predictive Model — Next-Day Profitability")

model_df = daily_account.copy()
model_df = model_df.sort_values(['account','date'])
model_df['next_day_profitable'] = model_df.groupby('account')['daily_pnl'].shift(-1).apply(lambda x: 1 if x > 0 else 0)
model_df['sentiment_num'] = model_df['sentiment'].map({'Fear': 0, 'Neutral': 1, 'Greed': 2})

features = ['num_trades','win_rate','avg_leverage','avg_size_usd','long_short_ratio','daily_pnl','sentiment_num']
model_df_clean = model_df[features + ['next_day_profitable']].dropna()

X = model_df_clean[features]
y = model_df_clean['next_day_profitable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nRandom Forest Accuracy: {acc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Profitable','Profitable']))

importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances.round(3))
importances.to_csv(f'{OUTPUTS}/feature_importances.csv')

# ── CHART 8: Feature Importance 
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Chart 8 — BONUS: Feature Importance for Profitability Prediction', fontweight='bold')
bar_colors = [GREED_COLOR if i < 3 else NEUTRAL_COLOR if i < 5 else FEAR_COLOR
              for i in range(len(importances))]
bars = ax.barh(importances.index, importances.values, color=bar_colors, alpha=0.85,
               edgecolor='white', linewidth=0.4)
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_xlabel('Feature Importance')
ax.set_title(f'RF Model Accuracy: {acc:.1%}')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{CHARTS}/chart8_feature_importance.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Chart 8 saved.")

print("\n" + "=" * 60)
print("ALL CHARTS AND OUTPUTS SAVED SUCCESSFULLY")
print("=" * 60)

# Print summary stats for notebook
print("\n--- FINAL SUMMARY STATS ---")
print("Fear days mean PnL:", round(perf[perf['sentiment']=='Fear']['daily_pnl'].mean(), 2))
print("Greed days mean PnL:", round(perf[perf['sentiment']=='Greed']['daily_pnl'].mean(), 2))
print("Fear win rate:", round(perf[perf['sentiment']=='Fear']['win_rate'].mean(), 3))
print("Greed win rate:", round(perf[perf['sentiment']=='Greed']['win_rate'].mean(), 3))
print("Model Accuracy:", round(acc, 3))
