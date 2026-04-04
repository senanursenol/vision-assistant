from fastapi import APIRouter, UploadFile, File
from app.domain.schemas.analyze_image import AnalyzeImageResponse
from app.services.analyze_service import AnalyzeService

router = APIRouter(prefix="/analyze", tags=["analyze"])

analyze_service = AnalyzeService()


@router.post("/image", response_model=AnalyzeImageResponse)
async def analyze_image(file: UploadFile = File(...)):
    content = await file.read()

    result = analyze_service.analyze_image(file.filename, content)

    return AnalyzeImageResponse(**result)