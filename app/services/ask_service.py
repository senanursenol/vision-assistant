from app.services.file_service import FileService
from app.services.detection_service import DetectionService
from app.services.ocr_service import OCRService
from app.services.vlm_service import VLMService
from app.services.tts_service import TTSService


class AskService:
    def __init__(self):
        self.file_service = FileService()
        self.detection_service = DetectionService()
        self.ocr_service = OCRService()
        self.vlm_service = VLMService()
        self.tts_service = TTSService()

    def ask_image(self, filename: str, content: bytes, question: str) -> dict:
        saved_path = self.file_service.save_file(filename, content)

        detections = self.detection_service.detect(saved_path)
        ocr_result = self.ocr_service.extract_text(saved_path)

        answer = self.vlm_service.answer_question(
            image_path=saved_path,
            question=question,
            detections=detections,
            ocr_result=ocr_result
        )

        audio = self.tts_service.text_to_speech(answer)

        return {
            "filename": filename,
            "content_type": "image",
            "saved_path": saved_path,
            "question": question,
            "answer": answer,
            "detections": detections,
            "ocr": ocr_result,
            "audio": audio,
            "message": "Görselle ilgili soru cevaplandı."
        }