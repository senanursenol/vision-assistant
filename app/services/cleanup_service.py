import os
import time
from pathlib import Path

class CleanupService:
    def __init__(self):
        # Temizlenecek hedef klasörler
        self.directories = [Path("data/inputs"), Path("data/outputs/audio")]
        # Dosyaların ne kadar süre tutulacağı (Örnek: 30 dakika = 1800 saniye)
        self.max_age_seconds = 1800 

    def cleanup_old_files(self):
        now = time.time()
        
        for directory in self.directories:
            if not directory.exists():
                continue

            for file_path in directory.iterdir():
                if file_path.is_file():
                    # Dosyanın oluşturulma/değiştirilme zamanını kontrol et
                    file_age = now - file_path.stat().st_mtime
                    
                    # Eğer dosya 30 dakikadan eskiyse, sil!
                    if file_age > self.max_age_seconds:
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Uyarı: {file_path} silinemedi - {e}")