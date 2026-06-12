import uuid
import subprocess
from pathlib import Path

class TTSService:
    def __init__(self):
        self.output_dir = Path("data/outputs/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Türkçe doğal ses modeli (Kadın sesi için: tr-TR-EmelNeural, Erkek sesi için: tr-TR-AhmetNeural)
        self.voice = "tr-TR-EmelNeural"

    def text_to_speech(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "audio_path": "",
                "audio_url": ""
            }

        filename = f"{uuid.uuid4()}.mp3"
        audio_path = self.output_dir / filename

        # Event-loop çakışmasını (500 Error) önlemek için işlemi arka planda CLI ile çalıştırıyoruz
        try:
            subprocess.run(
                [
                    "edge-tts", 
                    "--text", text, 
                    "--voice", self.voice, 
                    "--write-media", str(audio_path)
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"TTS Üretim Hatası: {e}")
            return {
                "audio_path": "",
                "audio_url": ""
            }

        return {
            "audio_path": str(audio_path),
            "audio_url": f"/outputs/audio/{filename}"
        }