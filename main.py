import uvicorn
from fastapi import FastAPI

from config import settings
from model_manager import ModelManager
from models import BusinessDescription, DomainResponse, HealthResponse
from routes import DomainRoutes

# Setup logging
settings.setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Domain Suggestion API",
    description="Generate domain name suggestions based on business description",
    version="1.0.0"
)

# Initialize model manager and routes
model_manager = ModelManager(
    model_name=settings.MODEL_NAME,
    model_path=settings.MODEL_PATH
)
route_handler = DomainRoutes(model_manager)


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts"""
    model_manager.load_model()


# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await route_handler.health_check()


@app.post("/generate-domains", response_model=DomainResponse)
async def generate_domains(request: BusinessDescription):
    """Generate domain suggestions endpoint"""
    return await route_handler.generate_domains(request)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT
    )
