from app.services.file_service import FileService
from app.services.detection_service import DetectionService
from app.services.ocr_service import OCRService
from app.services.vlm_service import VLMService
from app.services.tts_service import TTSService
from app.services.stt_service import STTService


class AskVoiceService:
    def __init__(self):
        self.file_service = FileService()
        self.detection_service = DetectionService()
        self.ocr_service = OCRService()
        self.vlm_service = VLMService()
        self.tts_service = TTSService()
        self.stt_service = STTService()

    def ask_with_voice(
        self,
        image_filename: str,
        image_content: bytes,
        audio_filename: str,
        audio_content: bytes
    ) -> dict:
        saved_image_path = self.file_service.save_file(image_filename, image_content)
        saved_audio_path = self.file_service.save_file(audio_filename, audio_content)

        question = self.stt_service.transcribe(saved_audio_path)

        detections = self.detection_service.detect(saved_image_path)
        ocr_result = self.ocr_service.extract_text(saved_image_path)

        answer = self.vlm_service.answer_question(
            image_path=saved_image_path,
            question=question,
            detections=detections,
            ocr_result=ocr_result
        )

        audio_answer = self.tts_service.text_to_speech(answer)

        return {
            "image_filename": image_filename,
            "audio_filename": audio_filename,
            "saved_image_path": saved_image_path,
            "saved_audio_path": saved_audio_path,
            "question": question,
            "answer": answer,
            "detections": detections,
            "ocr": ocr_result,
            "audio": audio_answer,
            "message": "Sesli soru cevaplandı."
        }