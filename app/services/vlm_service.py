class VLMService:
    def generate_description(self, detections: list, ocr_result: dict) -> str:
        labels = [d["label"] for d in detections]
        unique_labels = list(set(labels))
        ocr_text = ocr_result.get("text", "").strip()

        parts = []

        if unique_labels:
            parts.append(f"Algılanan nesneler: {', '.join(unique_labels)}.")

        if ocr_text:
            parts.append(f"Görünen yazı: {ocr_text}.")

        if not parts:
            return "Detaylı sahne yorumu üretilemedi."

        return "Detaylı sahne yorumu üretildi. " + " ".join(parts)