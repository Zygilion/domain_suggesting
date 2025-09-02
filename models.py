from typing import List

from pydantic import BaseModel, Field


class BusinessDescription(BaseModel):
    description: str = Field(
        max_length=1000,
        description="Business description to generate domain suggestions for"
    )


class DomainResponse(BaseModel):
    suggestions: List[str]
    processing_time: float
    business_description: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: float
