class RoutingService:
    def should_use_vlm(self, detections: list, ocr_result: dict) -> bool:
        if not detections:
            return True

        avg_conf = sum(d["confidence"] for d in detections) / len(detections)

        if avg_conf < 0.4:
            return True

        if len(detections) > 6:
            return True

        labels = [d["label"] for d in detections]

        critical_labels = {"person", "car", "bicycle", "bus", "truck", "motorcycle"}
        if any(label in critical_labels for label in labels):
            return False

        ocr_text = ocr_result.get("text", "").strip() if ocr_result else ""
        if len(ocr_text) > 30:
            return True

        return False