from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.domain.schemas.analyze_image import AnalyzeImageResponse
from app.services.analyze_service import AnalyzeService
from app.services.cleanup_service import CleanupService

router = APIRouter(prefix="/analyze", tags=["analyze"])

analyze_service = AnalyzeService()
cleanup_service = CleanupService()

@router.post("/image", response_model=AnalyzeImageResponse)
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    try:
        content = await file.read()
        
        # Analiz işlemini başlat
        result = analyze_service.analyze_image(file.filename, content)
        
        # İşlem başarılıysa arka planda eski dosyaları temizle
        background_tasks.add_task(cleanup_service.cleanup_old_files)
        
        return AnalyzeImageResponse(**result)
    
    except Exception as e:
        # Hata durumunda sistemi çökertmek yerine API hatası fırlat
        raise HTTPException(
            status_code=500, 
            detail=f"Görüntü analizinde beklenmeyen bir hata oluştu: {str(e)}"
        )