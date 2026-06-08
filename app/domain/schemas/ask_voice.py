from pydantic import BaseModel
from typing import Dict, List


class AskVoiceResponse(BaseModel):
    image_filename: str
    audio_filename: str
    saved_image_path: str
    saved_audio_path: str
    question: str
    answer: str
    detections: List[Dict]
    ocr: Dict
    audio: Dict
    message: str