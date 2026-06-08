from fastapi import APIRouter, UploadFile, File, Form

from app.domain.schemas.ask_image import AskImageResponse
from app.services.ask_service import AskService


router = APIRouter(prefix="/ask", tags=["ask"])

ask_service = AskService()


@router.post("/image", response_model=AskImageResponse)
async def ask_image(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    content = await file.read()

    result = ask_service.ask_image(
        filename=file.filename,
        content=content,
        question=question
    )

    return AskImageResponse(**result)