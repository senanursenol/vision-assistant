from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.domain.schemas.ask_voice import AskVoiceResponse
from app.services.ask_voice_service import AskVoiceService
from app.services.cleanup_service import CleanupService

router = APIRouter(prefix="/ask", tags=["ask"])

ask_voice_service = AskVoiceService()
cleanup_service = CleanupService()

@router.post("/voice", response_model=AskVoiceResponse)
async def ask_voice(
    background_tasks: BackgroundTasks,
    image_file: UploadFile = File(...),
    audio_file: UploadFile = File(...)
):
    try:
        image_content = await image_file.read()
        audio_content = await audio_file.read()

        result = ask_voice_service.ask_with_voice(
            image_filename=image_file.filename,
            image_content=image_content,
            audio_filename=audio_file.filename,
            audio_content=audio_content
        )

        # İşlem başarılıysa arka planda eski dosyaları temizle
        background_tasks.add_task(cleanup_service.cleanup_old_files)

        return AskVoiceResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Sesli soru işlenirken bir hata oluştu: {str(e)}"
        )