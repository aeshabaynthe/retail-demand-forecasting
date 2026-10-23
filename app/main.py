import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine

from app.schemas import PriceOptimizationRequest, OptimizationResponse, PricePointResult

# 1. Initialize FastAPI App
app = FastAPI(
    title="Retail Demand & Price Optimization API",
    description="Production API for demand forecasting, price elasticity, and profit maximization.",
    version="1.0.0"
)

# 2. Global Variables
MODEL = None
PREPROCESSOR = None
ENGINE = None

numerical_features = [
    "selling_price", "competitor_price", "discount_pct", "promotion_flag",
    "marketing_spend", "temperature_c", "humidity_pct", "inflation_pct",
    "interest_rate_pct", "consumer_confidence", "fuel_price", "gdp_growth_pct",
    "holiday_flag", "festival_flag", "week", "month", "quarter",
    "lag_1", "lag_2", "lag_4", "lag_52", "roll_mean_4", "roll_mean_13"
]
categorical_features = ["category", "channel", "region"]
all_features = numerical_features + categorical_features


@app.on_event("startup")
def load_artifacts():
    global MODEL, PREPROCESSOR, ENGINE
    load_dotenv()
    
    # Load Machine Learning Artifacts
    MODEL = joblib.load("models/xgb_pricing_model.joblib")
    PREPROCESSOR = joblib.load("models/preprocessor.joblib")
    
    # Connect to PostgreSQL
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    
    ENGINE = create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    print("FastAPI Engine initialized and model artifacts loaded successfully!")


@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": MODEL is not None}


@app.post("/optimize-price", response_model=OptimizationResponse)
def optimize_price(payload: PriceOptimizationRequest):
    # Query latest product observation from database
    query = f"""
        SELECT * 
        FROM retail.fact_demand_features 
        WHERE product_id = '{payload.product_id}' AND region = '{payload.region}'
        ORDER BY date DESC 
        LIMIT 1;
    """
    latest_df = pd.read_sql(query, ENGINE)
    
    if latest_df.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"No record found for Product '{payload.product_id}' in Region '{payload.region}'"
        )
    
    latest = latest_df.iloc[0]
    current_price = float(latest["selling_price"])
    unit_cost = float(latest["unit_cost"])
    list_price = float(latest["list_price"])
    
    # Generate candidate prices
    candidate_prices = np.arange(
        current_price * payload.min_price_factor,
        current_price * (payload.max_price_factor + 0.001),
        current_price * payload.step_factor
    )
    
    # Build counterfactual evaluation DataFrame
    scenario_df = pd.DataFrame({
        "selling_price": candidate_prices,
        "competitor_price": latest["competitor_price"],
        "discount_pct": ((list_price - candidate_prices) / list_price) * 100,
        "promotion_flag": latest["promotion_flag"],
        "marketing_spend": latest["marketing_spend"],
        "temperature_c": latest["temperature_c"],
        "humidity_pct": latest["humidity_pct"],
        "inflation_pct": latest["inflation_pct"],
        "interest_rate_pct": latest["interest_rate_pct"],
        "consumer_confidence": latest["consumer_confidence"],
        "fuel_price": latest["fuel_price"],
        "gdp_growth_pct": latest["gdp_growth_pct"],
        "holiday_flag": latest["holiday_flag"],
        "festival_flag": latest["festival_flag"],
        "week": latest["week"],
        "month": latest["month"],
        "quarter": latest["quarter"],
        "lag_1": latest["lag_1"],
        "lag_2": latest["lag_2"],
        "lag_4": latest["lag_4"],
        "lag_52": latest["lag_52"],
        "roll_mean_4": latest["roll_mean_4"],
        "roll_mean_13": latest["roll_mean_13"],
        "category": latest["category"],
        "channel": latest["channel"],
        "region": latest["region"]
    })
    
    # Predict demand & calculate unit economics
    X_encoded = PREPROCESSOR.transform(scenario_df[all_features])
    scenario_df["predicted_demand"] = MODEL.predict(X_encoded)
    scenario_df["revenue"] = scenario_df["selling_price"] * scenario_df["predicted_demand"]
    scenario_df["profit"] = (scenario_df["selling_price"] - unit_cost) * scenario_df["predicted_demand"]
    
    # Find profit-maximizing row
    optimal_idx = scenario_df["profit"].idxmax()
    optimal_row = scenario_df.loc[optimal_idx]
    
    # Format candidate price curve results
    curve_results = [
        PricePointResult(
            selling_price=round(row["selling_price"], 2),
            predicted_demand=round(row["predicted_demand"], 2),
            revenue=round(row["revenue"], 2),
            profit=round(row["profit"], 2)
        )
        for _, row in scenario_df.iterrows()
    ]
    
    return OptimizationResponse(
        product_id=payload.product_id,
        region=payload.region,
        current_price=round(current_price, 2),
        unit_cost=round(unit_cost, 2),
        optimal_price=round(optimal_row["selling_price"], 2),
        max_profit=round(optimal_row["profit"], 2),
        predicted_demand_at_optimal=round(optimal_row["predicted_demand"], 2),
        candidate_curve=curve_results
    )