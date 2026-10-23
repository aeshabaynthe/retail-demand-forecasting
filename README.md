# Retail Demand Forecasting & Price Optimization Engine

An end-to-end machine learning project that combines retail demand forecasting with price optimization. The system uses historical sales data and relevant market features to estimate demand under different pricing scenarios and identify price points that maximize expected profit.

The project combines exploratory data analysis, feature engineering, model training, PostgreSQL data storage, and a **FastAPI REST API** for serving price optimization predictions.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Example Request](#example-request)
- [Disclaimer](#disclaimer)

## Overview

Retail pricing decisions involve balancing selling price, customer demand, and profitability. This project explores how machine learning can support these decisions by evaluating multiple candidate prices and estimating their potential impact on demand, revenue, and profit.

The workflow consists of two main components:

- **Offline model pipeline:** Retrieves data from PostgreSQL, prepares features, trains the machine learning model, and exports the trained model and preprocessing artifacts.
- **Online API service:** Loads the saved artifacts and evaluates pricing scenarios through a FastAPI endpoint.

## Key Features

- **Data storage:** Uses PostgreSQL to store and retrieve the data required for model training.
- **Feature engineering:** Supports the preparation of time-series, calendar, promotional, and regional features, as applicable to the training dataset.
- **Demand prediction:** Uses an XGBoost regressor to estimate demand for candidate pricing scenarios.
- **Model persistence:** Exports trained model and preprocessing artifacts using Joblib.
- **Price optimization:** Evaluates candidate prices and compares predicted demand, revenue, and profit to identify the most profitable candidate.
- **REST API:** Exposes the optimization functionality through FastAPI with Pydantic request validation.
- **Interactive API documentation:** Provides Swagger UI for exploring and testing endpoints.

## System Architecture

```text
             PostgreSQL Database
                      |
                      v
              Offline Training
               export_models.py
                      |
                      v
          Trained Model & Preprocessor
                 (.joblib)
                      |
                      v
API Client ------> FastAPI Application
                      |
                      v
              Price Optimization
                      |
                      v
       Demand, Revenue & Profit Estimates
               and Optimal Price
```

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| API Framework | FastAPI, Uvicorn |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Model Persistence | Joblib |
| Database | PostgreSQL, SQLAlchemy |
| Configuration | python-dotenv |
| Analysis & Visualization | Jupyter Notebook, Statsmodels, Matplotlib, Seaborn |

## Project Structure

```text
retail-demand-forecasting/
├── app/
│   ├── main.py
│   └── schemas.py
├── data/
├── models/
│   ├── preprocessor.joblib
│   └── xgb_pricing_model.joblib
├── notebooks/
├── SQL/
├── export_models.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Getting Started

### 1. Prerequisites

Install the following before running the project:

- Python compatible with the project's dependencies
- PostgreSQL
- Git

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd retail-demand-forecasting
```

### 3. Install Dependencies

```bash
python -m venv .venv
```

Activate the virtual environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a local `.env` file in the project root. Configure it with the credentials and database name used by your PostgreSQL setup.

```dotenv
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=retail_db
```

Replace the example values with your actual local configuration. Ensure that `.env` is included in `.gitignore` and never commit database credentials.

### 5. Generate Model Artifacts

Run the model export script:

```bash
python export_models.py
```

This step is intended to retrieve the training data, fit the preprocessing pipeline and XGBoost model, and save the resulting artifacts for API inference.

Ensure that the database is running, the required tables are populated, and the database configuration matches the script before running it.

*If the trained model artifacts are already included in the repository, check the project's intended workflow before retraining them.*

### 6. Start the API Server

From the project root, run:

```bash
python -m uvicorn app.main:app --reload
```

The API will be available locally at:

`http://127.0.0.1:8000`

The `--reload` option is intended for development.

## API Documentation

Once the server is running, open the interactive Swagger documentation:

**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Checks API status and model-loading status. |
| `POST` | `/optimize-price` | Evaluates candidate prices and returns demand, revenue, and profit estimates. |

## Example Request

**Endpoint:** `POST /optimize-price`

**Request body:**

```json
{
  "product_id": "P001",
  "region": "Delhi NCR",
  "min_price_factor": 0.80,
  "max_price_factor": 1.20,
  "step_factor": 0.02
}
```

The request specifies a product, region, and price-search range relative to the current price. The exact accepted fields and constraints are defined by the API's request schema.

### Example Response

The following is an illustrative response showing the intended output structure. Actual values depend on the model, input data, and optimization logic.

```json
{
  "product_id": "P001",
  "region": "Delhi NCR",
  "current_price": 100.0,
  "unit_cost": 45.0,
  "optimal_price": 108.0,
  "max_profit": 2835.0,
  "predicted_demand_at_optimal": 45.0,
  "candidate_curve": [
    {
      "selling_price": 80.0,
      "predicted_demand": 62.5,
      "revenue": 5000.0,
      "profit": 2187.5
    },
    {
      "selling_price": 108.0,
      "predicted_demand": 45.0,
      "revenue": 4860.0,
      "profit": 2835.0
    }
  ]
}
```

The response is designed to provide:

- **Current and recommended price:** The existing price and the selected candidate price.
- **Demand estimate:** Predicted demand at the selected price.
- **Revenue and profit:** Estimated financial outcomes for candidate pricing scenarios.
- **Candidate curve:** Results across the evaluated price range, allowing comparison of alternative prices.

## Disclaimer

This project is developed for educational and portfolio purposes. Demand forecasts and price recommendations are estimates based on the available data and model assumptions. They should be validated against appropriate business constraints and real-world market conditions before being used for operational pricing decisions.

## Author

**Aesha Baynthe**

- [GitHub](<https://github.com/aeshabaynthe>)
- [LinkedIn](<www.linkedin.com/in/aeshabaynthe-dataanalyst>)
