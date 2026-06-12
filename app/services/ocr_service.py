import easyocr
import cv2


class OCRService:
    def __init__(self):
        # Türkçe + İngilizce
        self.reader = easyocr.Reader(['tr', 'en'], gpu=True)

    def extract_text(self, image_path: str) -> dict:
        # 1) image oku
        img = cv2.imread(image_path)

        if img is None:
            return {
                "text": "",
                "lines": []
            }

        # 2) grayscale (stabilite)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3) OCR
        results = self.reader.readtext(gray)

        texts = []
        lines = []

        for (bbox, text, confidence) in results:
            if confidence > 0.4:
                clean_text = text.strip()
                if clean_text:
                    texts.append(clean_text)
                    lines.append(clean_text)

        full_text = " ".join(texts)

        words = full_text.split()

        if len(words) <= 1:
            full_text = ""

        if len(full_text) < 5:
            full_text = ""

        return {
            "text": full_text,
            "lines": lines
        }