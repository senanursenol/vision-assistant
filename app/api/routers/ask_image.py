from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from app.domain.schemas.ask_image import AskImageResponse
from app.services.ask_service import AskService
from app.services.cleanup_service import CleanupService

router = APIRouter(prefix="/ask", tags=["ask"])

ask_service = AskService()
cleanup_service = CleanupService()

@router.post("/image", response_model=AskImageResponse)
async def ask_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        content = await file.read()

        result = ask_service.ask_image(
            filename=file.filename,
            content=content,
            question=question
        )

        # İşlem başarılıysa arka planda eski dosyaları temizle
        background_tasks.add_task(cleanup_service.cleanup_old_files)

        return AskImageResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Görselle ilgili soru cevaplanırken bir hata oluştu: {str(e)}"
        )