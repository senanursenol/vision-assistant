import torch
from faster_whisper import WhisperModel


class STTService:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        self.model = WhisperModel(
            "small",
            device=device,
            compute_type=compute_type
        )

    def transcribe(self, audio_path: str) -> str:
        segments, info = self.model.transcribe(
            audio_path,
            language="tr",
            beam_size=5
        )

        text_parts = []

        for segment in segments:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts).strip()