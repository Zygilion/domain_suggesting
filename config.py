import logging
import os


class Settings:
    """Application configuration settings"""

    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Model Settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "fine-tune-model/")

    # Generation Settings
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "500"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("TOP_P", "0.8"))
    TOP_K: int = int(os.getenv("TOP_K", "20"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def setup_logging(self):
        """Configure application logging"""
        logging.basicConfig(level=getattr(logging, self.LOG_LEVEL))


# Global settings instance
settings = Settings()
