from pydantic import BaseModel, Field
from typing import List

class PriceOptimizationRequest(BaseModel):
    product_id: str = Field(..., example="P001")
    region: str = Field(..., example="Delhi NCR")
    min_price_factor: float = Field(default=0.80, example=0.80)
    max_price_factor: float = Field(default=1.20, example=1.20)
    step_factor: float = Field(default=0.02, example=0.02)

class PricePointResult(BaseModel):
    selling_price: float
    predicted_demand: float
    revenue: float
    profit: float

class OptimizationResponse(BaseModel):
    product_id: str
    region: str
    current_price: float
    unit_cost: float
    optimal_price: float
    max_profit: float
    predicted_demand_at_optimal: float
    candidate_curve: List[PricePointResult]