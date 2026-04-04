class GuidanceService:
    def generate_basic_response(self, detections: list, ocr_result: dict) -> str:
        labels = [d["label"] for d in detections]
        unique_labels = list(set(labels))
        text = ocr_result.get("text", "").strip()

        parts = []

        if unique_labels:
            parts.append(f"Görüntüde algılanan nesneler: {', '.join(unique_labels)}.")

        if text:
            parts.append(f"Görünen yazı: {text}")

        if not parts:
            return "Görüntüde belirgin bir nesne veya yazı tespit edilemedi."

        return " ".join(parts)