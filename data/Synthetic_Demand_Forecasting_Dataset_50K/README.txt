Demand Forecasting & Pricing Strategy - Synthetic Dataset
===========================================================
Primary project input : Fact_Demand_Raw_50K.csv  (50,100 rows x 47 columns)
Reference/clean copy  : Fact_Demand_Clean_50K.csv (50,000 rows x 47 columns)

Grain: one row per Product x Region x Week (Monday). Regular weekly grid:
25 products x 8 regions x 250 weeks = 50,000 rows / 200 weekly time series.
Channel and Customer_Segment are fixed per Product-Region series.

Raw-file defects: 100 NULLs in each of Selling_Price, Competitor_Price,
Marketing_Spend, Temperature_C, Humidity_Pct; 100 exact duplicates; inconsistent
text (Brand case, Channel case/whitespace, Region case/'Bangalore' alias);
60 invalid Discount_Pct (negative or >60) - recompute from List/Selling price.

Suggested models: SARIMA (aggregate or per series, m=52) for demand;
Linear Regression + XGBoost for pricing (Demand_Units ~ price, competitor,
promo, marketing, weather, macro...).

Leakage warning: Units_Sold, Revenue, Gross_Margin, Inventory columns,
Stockout_Flag, Store_Traffic, Search_Index, Clicks, Conversions and Return_Units
are outcomes/co-movers of demand - do not use them as features for
same-week demand prediction. Build lag / rolling features yourself and use a
chronological train/test split.
