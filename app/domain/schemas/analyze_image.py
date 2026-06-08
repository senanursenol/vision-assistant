from pydantic import BaseModel
from typing import List, Dict


class AnalyzeImageResponse(BaseModel):
    filename: str
    content_type: str
    saved_path: str
    detections: List[Dict]
    ocr: Dict
    scene: str
    used_vlm: bool
    score: float
    audio: Dict
    message: str
    