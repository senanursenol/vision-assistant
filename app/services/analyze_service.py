from app.services.file_service import FileService
from app.services.detection_service import DetectionService
from app.services.ocr_service import OCRService
from app.services.guidance_service import GuidanceService
from app.services.vlm_service import VLMService


class AnalyzeService:
    def __init__(self):
        self.file_service = FileService()
        self.detection_service = DetectionService()
        self.ocr_service = OCRService()
        self.guidance_service = GuidanceService()
        self.vlm_service = VLMService()

    def analyze_image(self, filename: str, content: bytes) -> dict:
        saved_path = self.file_service.save_file(filename, content)

        detections = self.detection_service.detect(saved_path)
        ocr_result = self.ocr_service.extract_text(saved_path)

        labels = [d["label"] for d in detections]
        unique_labels = list(set(labels))
        ocr_text = ocr_result.get("text", "")

        detection_score = len(unique_labels)

        if detections:
            avg_conf = sum(d["confidence"] for d in detections) / len(detections)
        else:
            avg_conf = 0.0

        ocr_score = len(ocr_text.split()) if ocr_text else 0

        object_count = len(detections)
        
        complexity_score = (
            detection_score * 1.5 +
            object_count * 0.3 +
            ocr_score * 1.0 +
            avg_conf * 2.0
        )

        use_vlm = complexity_score > 5

        if use_vlm:
            scene = self.vlm_service.generate_description(detections, ocr_result)
        else:
            scene = self.guidance_service.generate_basic_response(detections, ocr_result)

        return {
            "filename": filename,
            "content_type": "image",
            "saved_path": saved_path,
            "detections": detections,
            "ocr": ocr_result,
            "scene": scene,
            "used_vlm": use_vlm,
            "score": round(complexity_score, 2),
            "message": "Image analiz edildi."
        }