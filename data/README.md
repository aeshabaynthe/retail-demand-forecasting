# Demand Forecasting & Pricing Strategy Model — Synthetic Data Generator

Generates a 50K-row weekly demand/pricing dataset for a project using **SARIMA** (demand forecasting) and **Linear Regression + XGBoost** (pricing strategy).

```bash
pip install numpy pandas
python generate_demand_dataset.py --seed 42 --out-dir data
```

## Dataset
| File | Rows | Cols | Notes |
|---|---|---|---|
| `Fact_Demand_Raw_50K.csv` | 50,100 | 47 | messy — main project input |
| `Fact_Demand_Clean_50K.csv` | 50,000 | 47 | reference / ground truth |
| `Dim_Product/Region/Date.csv` | 25 / 8 / 250 | — | dimensions |

**Grain:** Product × Region × Week (25 × 8 × 250 = 50,000), 2021-01-04 → 2025-10-13.

**Column groups (47):** identifiers & attributes · demand & sales · pricing & competitor · promotions · inventory & supply chain · marketing funnel · weather · macro indicators · calendar flags.

## Data-generating process
`log(demand) = level + trend + annual seasonality + festival/holiday lift + weather + macro + elasticity × log((1 − discount) / competitor_price_index) + marketing + AR(1) shock`

Category price elasticities range from −1.7 (Electronics) to −2.6 (Grocery).

## Deliberate raw-data defects
100 NULLs in each of `Selling_Price`, `Competitor_Price`, `Marketing_Spend`, `Temperature_C`, `Humidity_Pct` · 100 exact duplicates · inconsistent text in `Brand`, `Channel`, `Region` · 60 invalid `Discount_Pct` values.

## Modelling notes
- Use a chronological train/test split.
- Build lag / rolling / YoY features yourself.
- Don't use same-week outcomes (`Units_Sold`, `Revenue`, inventory, `Store_Traffic`, `Clicks`…) as features for demand.
