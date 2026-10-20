#!/usr/bin/env python3
"""
Synthetic dataset generator
Comprehensive Demand Forecasting and Pricing Strategy Model
===========================================================

Generates a weekly product x region panel with demand, pricing, competitor,
promotion, marketing, inventory, supply-chain, weather and macro columns.

Output (in --out-dir):
    Fact_Demand_Clean_50K.csv   50,000 rows x 47 cols, analysis-ready, sorted
    Fact_Demand_Raw_50K.csv     50,100 rows x 47 cols, deliberately messy
    Dim_Product.csv / Dim_Region.csv / Dim_Date.csv
    README.txt

Panel design
    25 products x 8 regions x 250 weeks = 50,000 rows (200 weekly time series)
    A regular weekly grid means every product-region series is a proper time
    series, so SARIMA (seasonal period 52) can be fitted without gaps.

Data-generating process (so models have real signal to recover)
    log(demand) = series level + trend + annual seasonality + festival/holiday
                  lift + weather effect + macro effect + price effect
                  + marketing effect + AR(1) shock
    price effect = elasticity * log( (1 - discount) / competitor_price_index )
                   -> demand reacts to price *relative to the competitor*.

Raw-file defects (for the data-cleaning stage):
    100 missing values in each of 5 columns, 100 exact duplicate rows,
    inconsistent text (Brand, Channel, Region), 60 invalid Discount_Pct values.

Usage:
    python generate_demand_dataset.py
    python generate_demand_dataset.py --seed 7 --out-dir data
"""
import argparse
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
N_PRODUCTS = 25
N_WEEKS = 250
START_DATE = "2021-01-04"  # a Monday; 250 weeks -> ends Oct 2025
N_DUPLICATES = 100
N_MISSING_PER_COL = 100
N_INVALID_DISCOUNT = 60

REGIONS = [
    # name, market_index, cluster, base_temp, temp_amp, monsoon_peak_doy, coastal
    ("Delhi NCR", 0.85, "North Metro", 25, 10, 215, 0),
    ("Mumbai", 1.20, "West Metro", 28, 3, 200, 1),
    ("Bengaluru", 1.25, "South Tech", 24, 3, 250, 0),
    ("Hyderabad", 1.15, "South Tech", 27, 6, 240, 0),
    ("Chennai", 1.10, "South Metro", 30, 3, 320, 1),
    ("Kolkata", 0.95, "East Metro", 27, 6, 210, 1),
    ("Pune", 1.05, "West Metro", 25, 5, 205, 0),
    ("Ahmedabad", 0.90, "West", 28, 7, 210, 0),
]

CATEGORY_CFG = {
    #                   price range       cost ratio    elasticity  weekly base  season amp / peak week
    "Grocery":         dict(price=(150, 1200),    cost=(0.70, 0.80), elast=-2.6, base=220, amp=0.06, peak=47,
                            temp=0.000, rain=0.000, fest=0.6, ret=0.005),
    "Personal Care":   dict(price=(250, 2500),    cost=(0.40, 0.55), elast=-2.2, base=150, amp=0.08, peak=40,
                            temp=0.004, rain=0.000, fest=0.8, ret=0.010),
    "Sports":          dict(price=(1500, 9000),   cost=(0.50, 0.62), elast=-2.0, base=60,  amp=0.15, peak=6,
                            temp=-0.006, rain=-0.003, fest=0.7, ret=0.020),
    "Home Appliances": dict(price=(4000, 18000),  cost=(0.60, 0.72), elast=-1.8, base=35,  amp=0.22, peak=22,
                            temp=0.025, rain=-0.001, fest=1.3, ret=0.025),
    "Electronics":     dict(price=(6000, 45000),  cost=(0.70, 0.82), elast=-1.7, base=25,  amp=0.10, peak=45,
                            temp=0.000, rain=0.000, fest=1.5, ret=0.030),
}
TIERS = ["Value", "Standard", "Premium"]
TIER_PRICE_MULT = {"Value": 0.80, "Standard": 1.00, "Premium": 1.30}
BRANDS = ["Zenora", "Kairo", "Veloxa", "Nimbus", "Solaris", "Trident"]
CHANNELS = ["Retail Store", "E-Commerce", "Marketplace", "B2B"]
CHANNEL_P = [0.40, 0.30, 0.20, 0.10]
CHANNEL_MULT = {"Retail Store": 1.00, "E-Commerce": 1.10, "Marketplace": 0.95, "B2B": 1.50}
STORE_TRAFFIC_FACTOR = {"Retail Store": 4.0, "E-Commerce": 0.3, "Marketplace": 0.3, "B2B": 0.6}
SEGMENTS_BY_TIER = {
    "Value": (["Budget", "Student", "Mass Market"], [0.4, 0.3, 0.3]),
    "Standard": (["Mass Market", "Family", "Student", "Business"], [0.4, 0.3, 0.2, 0.1]),
    "Premium": (["Premium", "Business", "Family"], [0.5, 0.25, 0.25]),
}
SEGMENT_MULT = {"Budget": 1.05, "Student": 0.95, "Mass Market": 1.10,
                "Family": 1.05, "Premium": 0.90, "Business": 1.00}

# Approximate real festival dates (week containing the date is flagged)
DIWALI = ["2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20"]
HOLI = ["2021-03-29", "2022-03-18", "2023-03-08", "2024-03-25", "2025-03-14"]
EID = ["2021-05-14", "2022-05-03", "2023-04-22", "2024-04-11", "2025-03-31"]

COLUMNS = [
    "Date", "Product_ID", "Region", "Category", "Brand", "Product_Tier", "Channel",
    "Customer_Segment", "Demand_Units", "Units_Sold", "Lost_Sales_Units", "List_Price",
    "Selling_Price", "Discount_Pct", "Promotion_Flag", "Competitor_Price",
    "Competitor_Price_Index", "Competitor_Promo_Flag", "Unit_Cost", "Revenue",
    "Gross_Margin", "Opening_Inventory", "Inventory_Received", "Closing_Inventory",
    "Reorder_Point", "Lead_Time_Days", "Stockout_Flag", "Return_Units", "Marketing_Spend",
    "Impressions", "Clicks", "Conversions", "Store_Traffic", "Search_Index",
    "Temperature_C", "Rainfall_mm", "Humidity_Pct", "Inflation_Pct", "Interest_Rate_Pct",
    "GDP_Growth_Pct", "Consumer_Confidence", "Fuel_Price", "Supplier_Reliability",
    "Supplier_Delay_Flag", "Campaign_ID", "Holiday_Flag", "Festival_Flag",
]


# --------------------------------------------------------------------------- #
# Dimension tables
# --------------------------------------------------------------------------- #
def make_dim_product(rng):
    cats = list(CATEGORY_CFG)
    rows = []
    for i in range(N_PRODUCTS):
        cat = cats[i % len(cats)]
        cfg = CATEGORY_CFG[cat]
        tier = rng.choice(TIERS, p=[0.30, 0.45, 0.25])
        base_price = rng.uniform(*cfg["price"]) * TIER_PRICE_MULT[tier]
        unit_cost = base_price * rng.uniform(*cfg["cost"])
        rows.append({
            "Product_ID": f"P{i + 1:03d}",
            "Product_Name": f"Product_{i + 1:02d}",
            "Category": cat,
            "Brand": rng.choice(BRANDS),
            "Product_Tier": tier,
            "Base_Price": round(base_price, 2),
            "Unit_Cost": round(unit_cost, 2),
            "Supplier_ID": f"S{rng.integers(101, 113)}",
        })
    return pd.DataFrame(rows)


def make_dim_region():
    return pd.DataFrame(
        [{"Region": r[0], "Market_Index": r[1], "Market_Cluster": r[2]} for r in REGIONS]
    )


def make_dim_date(dates):
    df = pd.DataFrame({"Date": dates})
    df["Week_Index"] = np.arange(1, len(df) + 1)
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["Month_Name"] = df.Date.dt.month_name()
    df["Quarter"] = "Q" + df.Date.dt.quarter.astype(str)
    df["Week"] = df.Date.dt.isocalendar().week.astype(int)
    df["Date"] = df.Date.dt.strftime("%Y-%m-%d")
    return df


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def mean_reverting(mean_path, kappa, sigma, rng):
    x = np.empty(len(mean_path))
    x[0] = mean_path[0]
    for t in range(1, len(x)):
        x[t] = x[t - 1] + kappa * (mean_path[t] - x[t - 1]) + sigma * rng.normal()
    return x


def week_flags(dates):
    """Festival lift (log scale), Festival_Flag and Holiday_Flag per week."""
    T = len(dates)
    lift = np.zeros(T)
    fest = np.zeros(T, dtype=int)
    hol = np.zeros(T, dtype=int)
    starts = dates.values

    def week_of(d):
        d = np.datetime64(pd.Timestamp(d))
        idx = np.searchsorted(starts, d, side="right") - 1
        return idx if 0 <= idx < T and d < starts[idx] + np.timedelta64(7, "D") else None

    for d in DIWALI:  # 3-week festive run-up, peaking in Diwali week
        i = week_of(d)
        if i is not None:
            for off, val in zip((-2, -1, 0), (0.12, 0.22, 0.30)):
                if 0 <= i + off < T:
                    lift[i + off] = max(lift[i + off], val)
                    fest[i + off] = 1
    for d, val in [(x, 0.10) for x in HOLI] + [(x, 0.08) for x in EID]:
        i = week_of(d)
        if i is not None:
            lift[i] = max(lift[i], val)
            fest[i] = 1
    for y in sorted(set(dates.year)):  # Christmas + national holidays
        for md, val, is_fest in [(f"{y}-12-25", 0.08, 1), (f"{y}-01-01", 0.0, 0),
                                 (f"{y}-01-26", 0.0, 0), (f"{y}-08-15", 0.0, 0),
                                 (f"{y}-10-02", 0.0, 0)]:
            i = week_of(md)
            if i is None:
                continue
            hol[i] = 1
            if is_fest:
                lift[i] = max(lift[i], val)
                fest[i] = 1
    return lift, fest, hol


# --------------------------------------------------------------------------- #
# Main generator
# --------------------------------------------------------------------------- #
def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")
    T = N_WEEKS
    wk = np.arange(T)
    woy = dates.isocalendar().week.values.astype(int)
    doy = dates.dayofyear.values
    years = wk / 52.0

    dim_product = make_dim_product(rng)
    dim_region = make_dim_region()
    dim_date = make_dim_date(dates)
    P, R = len(dim_product), len(REGIONS)
    S = P * R

    # ---- national macro series (T,) ------------------------------------- #
    yr_pts = [0, 1, 2, 3, 4.8]
    inflation = mean_reverting(np.interp(years, yr_pts, [5.0, 6.2, 5.6, 4.8, 4.4]), 0.08, 0.10, rng)
    interest = mean_reverting(np.interp(years, yr_pts, [4.0, 4.6, 6.3, 6.5, 6.2]), 0.08, 0.04, rng)
    gdp = mean_reverting(np.interp(years, yr_pts, [7.0, 7.2, 7.5, 6.8, 6.5]), 0.05, 0.06, rng)
    conf = mean_reverting(np.interp(years, yr_pts, [95, 98, 102, 104, 103]), 0.06, 0.6, rng)
    fuel = mean_reverting(np.interp(years, yr_pts, [85, 95, 100, 102, 100]), 0.06, 0.6, rng)

    lift, fest_flag, hol_flag = week_flags(dates)

    # ---- regional weather (R, T) ---------------------------------------- #
    temp = np.empty((R, T)); rain = np.empty((R, T)); hum = np.empty((R, T))
    for r, (_, _, _, b_t, amp, m_peak, coastal) in enumerate(REGIONS):
        temp[r] = (b_t + amp * np.cos(2 * np.pi * (doy - 150) / 365)
                   + mean_reverting(np.zeros(T), 0.5, 1.0, rng))
        m = np.exp(-(((doy - m_peak + 182) % 365 - 182) / 40.0) ** 2)  # monsoon intensity 0..1
        wet = rng.random(T) < (0.12 + 0.68 * m)
        rain[r] = wet * rng.gamma(1.5, 22 * (0.4 + 1.6 * m))
        hum[r] = np.clip(42 + 10 * coastal + 32 * m + rng.normal(0, 5, T), 20, 99)

    # ---- series attributes (S,) ----------------------------------------- #
    prod_idx = np.repeat(np.arange(P), R)
    reg_idx = np.tile(np.arange(R), P)
    prod = dim_product.iloc[prod_idx].reset_index(drop=True)
    cfg = [CATEGORY_CFG[c] for c in prod.Category]
    market = np.array([REGIONS[r][1] for r in reg_idx])

    channel = rng.choice(CHANNELS, size=S, p=CHANNEL_P)
    segment = np.array([rng.choice(SEGMENTS_BY_TIER[t][0], p=SEGMENTS_BY_TIER[t][1])
                        for t in prod.Product_Tier])
    prod_scale = np.repeat(rng.lognormal(0, 0.30, P), R)
    level = (np.array([c["base"] for c in cfg]) * prod_scale * market
             * np.array([CHANNEL_MULT[c] for c in channel])
             * np.array([SEGMENT_MULT[s] for s in segment]) * rng.lognormal(0, 0.10, S))
    elast = np.array([c["elast"] for c in cfg]) + rng.normal(0, 0.20, S)
    growth = np.repeat(rng.normal(0.05, 0.02, P), R)  # annual log-growth per product

    supplier_rel_base = {s: rng.uniform(0.80, 0.97) for s in dim_product.Supplier_ID.unique()}
    rel_s = np.array([supplier_rel_base[s] for s in prod.Supplier_ID])
    lt_base = np.repeat(rng.uniform(3, 16, P), R)
    ctr = rng.uniform(0.02, 0.05, S)
    cvr = rng.uniform(0.05, 0.13, S)
    spend_by_cat = {"Grocery": 800, "Personal Care": 1200, "Sports": 1500,
                    "Home Appliances": 2200, "Electronics": 3000}
    spend_base = np.array([spend_by_cat[c] for c in prod.Category], dtype=float) * market

    # ---- time-varying panel (S, T) -------------------------------------- #
    festival = np.tile(fest_flag, (S, 1))
    holiday = np.tile(hol_flag, (S, 1))
    lift_st = np.outer([c["fest"] for c in cfg], lift)

    # competitor promo & own promo
    comp_promo = (rng.random((S, T)) < 0.07).astype(int)
    promo = (rng.random((S, T)) < (0.13 + 0.14 * festival)).astype(int)
    disc = np.where(promo == 1,
                    10 + 20 * rng.beta(2, 2.5, (S, T)),          # promo depth 10-30 %
                    rng.triangular(0, 2.5, 7, (S, T)))            # everyday 0-7 %
    disc = np.round(disc, 2)
    cpi = np.clip(rng.normal(1.01, 0.03, (S, T)) - 0.09 * comp_promo, 0.85, 1.12)
    cpi = np.round(cpi, 3)

    price_drift = np.exp(0.03 * years)[None, :]                   # ~3 % / year list-price drift
    list_price = prod.Base_Price.values[:, None] * price_drift * rng.lognormal(0, 0.01, (S, T))
    list_price = np.round(list_price, 2)
    selling = np.round(list_price * (1 - disc / 100), 2)
    comp_price = np.round(list_price * cpi, 2)
    cost = (prod.Unit_Cost.values[:, None] * np.exp(0.03 * years)[None, :]
            * (1 + 0.004 * (fuel - 95))[None, :] * rng.lognormal(0, 0.008, (S, T)))
    cost = np.round(cost, 2)

    # marketing funnel
    spend = (spend_base[:, None] * (1 + 0.9 * promo + 0.5 * festival)
             * rng.lognormal(0, 0.25, (S, T)))
    spend = np.round(np.clip(spend, 100, 8000), 2)
    impressions = np.round(spend * 35 * rng.lognormal(0, 0.25, (S, T))).astype(int)
    clicks = np.round(impressions * ctr[:, None] * rng.lognormal(0, 0.10, (S, T))).astype(int)
    conv = np.round(clicks * cvr[:, None] * rng.lognormal(0, 0.10, (S, T))).astype(int)

    # weather mapped to series
    temp_st, rain_st, hum_st = temp[reg_idx], rain[reg_idx], hum[reg_idx]

    # ---- demand ---------------------------------------------------------- #
    season = np.array([c["amp"] for c in cfg])[:, None] * np.cos(
        2 * np.pi * (woy[None, :] - np.array([c["peak"] for c in cfg])[:, None]) / 52.0)
    weather = (np.array([c["temp"] for c in cfg])[:, None] * (temp_st - 28)
               + np.array([c["rain"] for c in cfg])[:, None] * rain_st)
    macro = (0.012 * (conf - 100) - 0.03 * (inflation - 5) - 0.02 * (interest - 6)
             + 0.02 * (gdp - 6.8))[None, :]
    price_eff = elast[:, None] * np.log((1 - disc / 100) / cpi)
    mkt_eff = 0.06 * np.log1p(spend / 1500)
    shock = np.empty((S, T)); shock[:, 0] = rng.normal(0, 0.08, S)
    for t in range(1, T):                                          # AR(1) momentum
        shock[:, t] = 0.5 * shock[:, t - 1] + rng.normal(0, 0.08, S)

    log_lam = (np.log(level)[:, None] + growth[:, None] * years[None, :] + season + lift_st
               + 0.03 * holiday + weather + macro + price_eff + 0.08 * promo + mkt_eff + shock)
    lam = np.exp(log_lam)
    demand = np.maximum(1, rng.poisson(lam * rng.lognormal(0, 0.06, (S, T)))).astype(int)

    # ---- supply chain & inventory --------------------------------------- #
    reliability = np.clip(rel_s[:, None] + rng.normal(0, 0.02, (S, T)), 0.75, 0.99)
    delay = (rng.random((S, T)) > reliability).astype(int)
    lead = np.clip(np.round(lt_base[:, None] + rng.normal(0, 1.2, (S, T)) + 4 * delay), 2, 25).astype(int)
    lvl_t = level[:, None] * np.exp(growth[:, None] * years[None, :])
    rop = np.ceil(lvl_t * (lead / 7.0 + 0.6)).astype(int)
    up_to = rop + np.ceil(lvl_t * 1.5).astype(int)

    opening = np.zeros((S, T), int); received = np.zeros((S, T), int)
    closing = np.zeros((S, T), int); sold = np.zeros((S, T), int)
    stock = up_to[:, 0].copy()
    for t in range(T):
        opening[:, t] = stock
        order = np.where(stock < rop[:, t], up_to[:, t] - stock, 0)
        received[:, t] = np.round(order * (1 - 0.5 * delay[:, t])).astype(int)
        avail = stock + received[:, t]
        sold[:, t] = np.minimum(demand[:, t], avail)
        closing[:, t] = avail - sold[:, t]
        stock = closing[:, t]
    lost = demand - sold
    stockout = (lost > 0).astype(int)

    # ---- outcomes & funnel ---------------------------------------------- #
    revenue = np.round(sold * selling, 2)
    margin = np.round(revenue - sold * cost, 2)
    ret_rate = np.array([c["ret"] for c in cfg])[:, None]
    returns = rng.poisson(sold * ret_rate * rng.lognormal(0, 0.2, (S, T)))
    tf = np.array([STORE_TRAFFIC_FACTOR[c] for c in channel])[:, None]
    traffic = np.round(demand * tf * rng.uniform(0.7, 1.3, (S, T))).astype(int)
    rel_demand = demand / level[:, None]
    search = np.round(100 * rel_demand ** 0.5 * (1 + lift) * rng.lognormal(0, 0.08, (S, T)), 2)
    camp_id = np.where(promo == 1, ((wk // 8) % 30 + 1)[None, :].repeat(S, 0), 0)

    # ---- assemble long table -------------------------------------------- #
    def flat(a):
        return np.asarray(a).reshape(-1)

    def per_series(v):
        return np.repeat(np.asarray(v), T)

    def per_time(v):
        return np.tile(np.asarray(v), S)

    df = pd.DataFrame({
        "Date": per_time(dates.strftime("%Y-%m-%d")),
        "Product_ID": per_series(prod.Product_ID),
        "Region": per_series([REGIONS[r][0] for r in reg_idx]),
        "Category": per_series(prod.Category),
        "Brand": per_series(prod.Brand),
        "Product_Tier": per_series(prod.Product_Tier),
        "Channel": per_series(channel),
        "Customer_Segment": per_series(segment),
        "Demand_Units": flat(demand),
        "Units_Sold": flat(sold),
        "Lost_Sales_Units": flat(lost),
        "List_Price": flat(list_price),
        "Selling_Price": flat(selling),
        "Discount_Pct": flat(disc),
        "Promotion_Flag": flat(promo),
        "Competitor_Price": flat(comp_price),
        "Competitor_Price_Index": flat(cpi),
        "Competitor_Promo_Flag": flat(comp_promo),
        "Unit_Cost": flat(cost),
        "Revenue": flat(revenue),
        "Gross_Margin": flat(margin),
        "Opening_Inventory": flat(opening),
        "Inventory_Received": flat(received),
        "Closing_Inventory": flat(closing),
        "Reorder_Point": flat(rop),
        "Lead_Time_Days": flat(lead),
        "Stockout_Flag": flat(stockout),
        "Return_Units": flat(returns),
        "Marketing_Spend": flat(spend),
        "Impressions": flat(impressions),
        "Clicks": flat(clicks),
        "Conversions": flat(conv),
        "Store_Traffic": flat(traffic),
        "Search_Index": flat(search),
        "Temperature_C": np.round(flat(temp_st), 2),
        "Rainfall_mm": np.round(flat(rain_st), 2),
        "Humidity_Pct": np.round(flat(hum_st), 2),
        "Inflation_Pct": per_time(np.round(inflation, 2)),
        "Interest_Rate_Pct": per_time(np.round(interest, 2)),
        "GDP_Growth_Pct": per_time(np.round(gdp, 2)),
        "Consumer_Confidence": per_time(np.round(conf, 2)),
        "Fuel_Price": per_time(np.round(fuel, 2)),
        "Supplier_Reliability": np.round(flat(reliability), 3),
        "Supplier_Delay_Flag": flat(delay),
        "Campaign_ID": flat(camp_id),
        "Holiday_Flag": flat(holiday),
        "Festival_Flag": flat(festival),
    })[COLUMNS]
    return df, dim_product, dim_region, dim_date


# --------------------------------------------------------------------------- #
# Raw (dirty) version
# --------------------------------------------------------------------------- #
def make_raw(clean, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    raw = clean.copy()
    n = len(raw)

    # 1) inconsistent text
    m = rng.random(n) < 0.08
    raw.loc[m, "Brand"] = raw.loc[m, "Brand"].str.upper()
    u = rng.random(n)
    ch = raw["Channel"].copy()
    ch[u < 0.10] = ch[u < 0.10].str.lower()
    pad = (u >= 0.10) & (u < 0.16)
    ch[pad] = " " + ch[pad] + " "
    raw["Channel"] = ch
    u = rng.random(n)
    reg = raw["Region"].copy()
    reg[(u < 0.02)] = reg[(u < 0.02)].str.lower()
    reg[(raw["Region"] == "Bengaluru") & (u > 0.985)] = "Bangalore"
    raw["Region"] = reg

    # 2) missing values (100 per column)
    for col in ["Selling_Price", "Competitor_Price", "Marketing_Spend",
                "Temperature_C", "Humidity_Pct"]:
        raw.loc[rng.choice(n, N_MISSING_PER_COL, replace=False), col] = np.nan

    # 3) invalid discounts (negative or > 60 %); Selling_Price stays correct
    bad = rng.choice(n, N_INVALID_DISCOUNT, replace=False)
    half = N_INVALID_DISCOUNT // 2
    raw.loc[bad[:half], "Discount_Pct"] = np.round(-rng.uniform(1, 5, half), 2)
    raw.loc[bad[half:], "Discount_Pct"] = np.round(rng.uniform(61, 125, N_INVALID_DISCOUNT - half), 2)

    # 4) exact duplicates, then shuffle everything
    dup = raw.iloc[rng.choice(n, N_DUPLICATES, replace=False)]
    raw = pd.concat([raw, dup], ignore_index=True)
    return raw.sample(frac=1, random_state=seed).reset_index(drop=True)


README = """Demand Forecasting & Pricing Strategy - Synthetic Dataset
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
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default="synthetic_demand_data")
    args = ap.parse_args()

    clean, dim_p, dim_r, dim_d = generate(args.seed)
    raw = make_raw(clean, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    clean.to_csv(os.path.join(args.out_dir, "Fact_Demand_Clean_50K.csv"), index=False)
    raw.to_csv(os.path.join(args.out_dir, "Fact_Demand_Raw_50K.csv"), index=False)
    dim_p.to_csv(os.path.join(args.out_dir, "Dim_Product.csv"), index=False)
    dim_r.to_csv(os.path.join(args.out_dir, "Dim_Region.csv"), index=False)
    dim_d.to_csv(os.path.join(args.out_dir, "Dim_Date.csv"), index=False)
    with open(os.path.join(args.out_dir, "README.txt"), "w") as f:
        f.write(README)

    print(f"clean: {clean.shape} | raw: {raw.shape} | seed={args.seed}")
    print(f"stockout rate: {clean.Stockout_Flag.mean():.2%} | "
          f"promo rate: {clean.Promotion_Flag.mean():.2%} | "
          f"date range: {clean.Date.min()} -> {clean.Date.max()}")
    print(f"files written to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
