import logging
import time

from fastapi import HTTPException

from model_manager import ModelManager
from models import BusinessDescription, DomainResponse, HealthResponse

logger = logging.getLogger(__name__)


class DomainRoutes:
    """Route handlers for domain generation API"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    async def health_check(self) -> HealthResponse:
        """Health check endpoint to verify API and model status"""
        return HealthResponse(
            status="healthy" if self.model_manager.is_ready else "unhealthy",
            model_loaded=self.model_manager.is_ready,
            timestamp=time.time()
        )

    async def generate_domains(self, request: BusinessDescription) -> DomainResponse:
        """Generate domain suggestions based on business description"""

        if not self.model_manager.is_ready:
            raise HTTPException(status_code=503, detail="Model not loaded. Please check the health endpoint.")

        try:
            logger.info(f"Generating domains for: {request.description}")

            suggestions, processing_time = self.model_manager.generate_domains(
                request.description
            )

            logger.info(f"Generated {len(suggestions)} suggestions in {processing_time:.2f}s")

            return DomainResponse(
                suggestions=suggestions,
                processing_time=processing_time,
                business_description=request.description
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
