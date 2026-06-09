from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routers import health, analyze_image, ask_image, ask_voice
from app.core.config import settings


Path("data/outputs").mkdir(parents=True, exist_ok=True)
Path("app/static").mkdir(parents=True, exist_ok=True)

app = FastAPI(title=settings.APP_NAME)

app.mount("/outputs", StaticFiles(directory="data/outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def read_index():
    return FileResponse("app/static/index.html")


app.include_router(health.router)
app.include_router(analyze_image.router)
app.include_router(ask_image.router)
app.include_router(ask_voice.router)