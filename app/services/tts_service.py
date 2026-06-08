import uuid
from pathlib import Path
from gtts import gTTS


class TTSService:
    def __init__(self):
        self.output_dir = Path("data/outputs/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def text_to_speech(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "audio_path": "",
                "audio_url": ""
            }

        filename = f"{uuid.uuid4()}.mp3"
        audio_path = self.output_dir / filename

        tts = gTTS(text=text, lang="tr", slow=False)
        tts.save(str(audio_path))

        return {
            "audio_path": str(audio_path),
            "audio_url": f"/outputs/audio/{filename}"
        }