DROP SCHEMA retail CASCADE;
DROP TABLE IF EXISTS retail.fact_demand CASCADE;
CREATE SCHEMA IF NOT EXISTS retail;
CREATE TABLE retail.dim_product (
    Product_ID VARCHAR(20) PRIMARY KEY,
    Brand VARCHAR(100),
    Category VARCHAR(100),
	base_price INT,
	unit_cost INT
);
CREATE TABLE retail.dim_region (
    region VARCHAR(20) PRIMARY KEY,
    market_index VARCHAR(100),
    market_cluster VARCHAR(100)
);
CREATE TABLE retail.dim_date (
    Date DATE PRIMARY KEY,
	week_index INT,
    year INT,
    month INT,
	month_name VARCHAR,
    quarter VARCHAR,
	week INT
);
CREATE TABLE retail.fact_demand (
    product_id VARCHAR(20) NOT NULL,
    Date DATE NOT NULL,
    Region VARCHAR(100) NOT NULL,
	category VARCHAR(100),
	brand VARCHAR(100),
	product_tier VARCHAR(20),
    Channel VARCHAR(100),
	customer_segment VARCHAR(100),
	demand_units NUMERIC,
    units_sold NUMERIC,
    lost_sales_units NUMERIC,
    list_price NUMERIC,
    selling_price NUMERIC,
	discount_pct NUMERIC,
	promotion_flag INT,
    competitor_price NUMERIC,
    competitor_price_index NUMERIC,
	competitor_promo_flag INT,
    unit_cost NUMERIC,
    revenue NUMERIC,
    gross_margin NUMERIC,
    opening_inventory NUMERIC,
	inventory_recieved NUMERIC,
    closing_inventory NUMERIC,
	reorder_point NUMERIC,
    lead_time_days NUMERIC,
	stockout_flag INT,
    return_unit INT,
    marketing_spend NUMERIC,
	impressions BIGINT,
	clicks INT,
	conversions NUMERIC,
	store_traffic NUMERIC,
	search_index NUMERIC,
    temperature_c NUMERIC,
	rainfall_mm NUMERIC,
    humidity_pct NUMERIC,
    inflation_pct NUMERIC,
    interest_rate_pct NUMERIC,
	gdp_growth_pct NUMERIC,
    consumer_confidence NUMERIC,
    fuel_price NUMERIC,
	supplier_reliability NUMERIC,
	supplier_delay_flag NUMERIC,
	campaign_id NUMERIC,
	holiday_flag INT,
	festival_flag INT
);
CREATE INDEX idx_fact_date
ON retail.fact_demand(date);

CREATE INDEX idx_fact_product_region_date
ON retail.fact_demand(product_id, region, date);

CREATE INDEX idx_fact_category
ON retail.fact_demand(product_id);