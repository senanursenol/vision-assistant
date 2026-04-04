import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "Vision Assistant API")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT: int = int(os.getenv("APP_PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()