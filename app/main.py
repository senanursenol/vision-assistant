from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from contextlib import asynccontextmanager
from app.services.vlm_service import vlm_service

from app.api.routers import health, analyze_image, ask_image, ask_voice
from app.core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Sistem başlatılıyor: VLM Modeli GPU'ya yükleniyor (Lütfen bekleyin)...")
    vlm_service._load_model()
    print("✅ Model başarıyla VRAM'e yüklendi! Sistem isteklere hazır.")
    yield
    print("🛑 Sistem kapanıyor. Kaynaklar serbest bırakılıyor...")

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

if _name_ == "_main_":
    uvicorn.run("app:app", host=settings.APP_HOST, port=settings.APP_PORT)