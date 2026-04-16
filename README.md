# 📊 Trader Performance vs Market Sentiment
### Primetrade.ai — Data Science Intern Assignment

> **Analyze how Bitcoin Fear/Greed sentiment relates to trader behavior and performance on Hyperliquid, uncovering patterns for smarter trading strategies.**

---

## 🗂️ Repository Structure

```
primetrade_analysis/
├── trader_sentiment_analysis.ipynb   ← Main analysis notebook
├── README.md                          ← This file
├── charts/
│   ├── chart1_performance_fear_greed.png   ← PnL / Win Rate distributions
│   ├── chart2_behavior_fear_greed.png      ← Trade frequency, leverage, size
│   ├── chart3_segmentation.png             ← 3 trader segments
│   ├── chart4_insight_time_pnl.png         ← PnL over time by sentiment
│   ├── chart5_leverage_pnl.png             ← Leverage & size vs PnL scatter
│   ├── chart6_trading_bias.png             ← Long/short ratio heatmap
│   ├── chart7_clusters.png                 ← K-Means behavioral archetypes
│   └── chart8_feature_importance.png       ← RF model feature importance
└── outputs/
    ├── daily_account_metrics.csv       ← Daily PnL, win rate, leverage per trader
    ├── fear_greed_performance.csv      ← Aggregated stats: Fear vs Greed
    ├── behavior_by_sentiment.csv       ← Behavioral metrics by sentiment
    ├── account_segments.csv            ← Full per-account segment labels
    ├── cluster_summary.csv             ← K-Means cluster profiles
    └── feature_importances.csv         ← RF model feature importances
```

---

## ⚙️ Setup & How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Data Files
Place the two CSVs in the same directory as the notebook:
- `fear_greed_index.csv` — Bitcoin Fear/Greed Index
- `historical_data.csv` — Hyperliquid trader history

### Run the Notebook
```bash
jupyter notebook trader_sentiment_analysis.ipynb
```

Or run as a script:
```bash
python analysis.py
```

---

## 📈 Datasets

| Dataset | Rows | Columns | Date Range |
|---|---|---|---|
| Bitcoin Fear/Greed Index | 2,644 | 4 | 2018-02-01 → 2025-05-02 |
| Hyperliquid Trader History | 211,224 | 16 | 2023-05-01 → 2025-05-01 |
| **Merged (overlap)** | **~211K** | — | **2023-05-01 → 2025-05-01** |

---

##  Key Findings

### 1. Performance — Fear vs Greed Days
| Metric | Fear | Greed |
|---|---|---|
| Mean Daily PnL | **$7,055** | $5,581 |
| Median Daily PnL | $635 | **$842** |
| Win Rate | 83.2% | **83.7%** |
| Avg Max Drawdown | **-$16,263** | -$24,091 |

**→ Fear days yield higher mean PnL and lower drawdown risk. Greed days are more volatile.**

### 2. Behavioral Changes
| Behavior | Fear | Greed |
|---|---|---|
| Avg Trades/Day | **70.5** | 54.1 |
| Avg Leverage | **27.3×** | 25.6× |
| Avg Position Size | **$11,597** | $6,718 |
| Long/Short Ratio | **36.4** | 11.8 |

**→ Traders are 30% more active and 72% more aggressive in Fear — a contrarian dip-buying pattern.**

### 3. Trader Segmentation
- **High Leverage traders** earn 2× PnL of Low Leverage (with only marginal win rate reduction)
- **Consistent Winners** have 94% win rate, 9.85 trades/day — statistically separable from losers
- **K-Means clustering** reveals 4 archetypes: High Earners, Moderate, Low Earners, Struggling

---

## 🎯 Strategy Recommendations

### Strategy 1 — "Buy Fear, Respect Greed"
- **Fear days:** Increase long bias; maintain or slightly raise position size; leverage is acceptable (drawdown lower)
- **Greed days:** Reduce position size by 30–40%; cap leverage; take profits earlier

### Strategy 2 — Segment-Aware Sentiment Scaling
| Segment | Fear Rule | Greed Rule |
|---|---|---|
| Consistent Winners | Trade normally or increase frequency | Reduce leverage 15% |
| High Leverage Traders | OK — maintain leverage | **Cap at 20×** |
| Frequent Traders | Increase frequency | Cut trades by 20% |
| Consistent Losers | **Do not trade** | Minimum size only |

---

##  Bonus: Predictive Model

**Random Forest Classifier** — Predicts next-day profitability (binary)

| Metric | Value |
|---|---|
| Accuracy | **86%** |
| Top Feature | Current-day PnL (importance: 0.256) |
| Sentiment contribution | 3.2% (useful, not dominant) |

Behavioral features (win rate, size, leverage) are stronger predictors of next-day profitability than sentiment alone.

---

## 📌 Methodology Notes

1. **Sentiment simplification:** Extreme Fear + Fear → "Fear"; Extreme Greed + Greed → "Greed"; Neutral stays
2. **Leverage proxy:** `Size USD / |Start Position|` (clipped 1–100×)
3. **Closing trades only** used for realized PnL analysis (avoids double-counting open/close)
4. **Net PnL = Closed PnL − Fee** for accuracy
5. **Date alignment** done at daily granularity (timezone: IST → UTC normalized)
