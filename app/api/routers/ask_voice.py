from fastapi import APIRouter, UploadFile, File

from app.domain.schemas.ask_voice import AskVoiceResponse
from app.services.ask_voice_service import AskVoiceService


router = APIRouter(prefix="/ask", tags=["ask"])

ask_voice_service = AskVoiceService()


@router.post("/voice", response_model=AskVoiceResponse)
async def ask_voice(
    image_file: UploadFile = File(...),
    audio_file: UploadFile = File(...)
):
    image_content = await image_file.read()
    audio_content = await audio_file.read()

    result = ask_voice_service.ask_with_voice(
        image_filename=image_file.filename,
        image_content=image_content,
        audio_filename=audio_file.filename,
        audio_content=audio_content
    )

    return AskVoiceResponse(**result)