from fastapi import FastAPI
from app.api.routers import health, analyze_image
from app.core.config import settings

app = FastAPI(title=settings.APP_NAME)

app.include_router(health.router)
app.include_router(analyze_image.router)