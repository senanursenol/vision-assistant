from pydantic import BaseModel
from typing import Dict, List


class AskImageResponse(BaseModel):
    filename: str
    content_type: str
    saved_path: str
    question: str
    answer: str
    detections: List[Dict]
    ocr: Dict
    audio: Dict
    message: str