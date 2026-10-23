import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Fetching demand features from PostgreSQL...")
query = "SELECT * FROM retail.fact_demand_features ORDER BY date, product_id, region;"
df = pd.read_sql(query, engine)

numerical_features = [
    "selling_price", "competitor_price", "discount_pct", "promotion_flag",
    "marketing_spend", "temperature_c", "humidity_pct", "inflation_pct",
    "interest_rate_pct", "consumer_confidence", "fuel_price", "gdp_growth_pct",
    "holiday_flag", "festival_flag", "week", "month", "quarter",
    "lag_1", "lag_2", "lag_4", "lag_52", "roll_mean_4", "roll_mean_13"
]
categorical_features = ["category", "channel", "region"]
all_features = numerical_features + categorical_features

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ],
    remainder="passthrough"
)

X = preprocessor.fit_transform(df[all_features])
y = df["demand_units"]

print("Fitting XGBoost Pricing Model...")
xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model, "models/xgb_pricing_model.joblib")
joblib.dump(preprocessor, "models/preprocessor.joblib")

print("Model and Preprocessor successfully saved to 'models/' directory!")