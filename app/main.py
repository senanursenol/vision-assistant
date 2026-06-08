from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routers import health, analyze_image
from app.core.config import settings


Path("data/outputs").mkdir(parents=True, exist_ok=True)

app = FastAPI(title=settings.APP_NAME)

app.mount("/outputs", StaticFiles(directory="data/outputs"), name="outputs")

app.include_router(health.router)
app.include_router(analyze_image.router)