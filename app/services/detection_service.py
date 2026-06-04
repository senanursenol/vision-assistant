from ultralytics import YOLO


class DetectionService:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, image_path: str) -> list:
        results = self.model(image_path, device=0)

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    {
                        "label": self.model.names[class_id],
                        "confidence": round(confidence, 4),
                        "bbox": [
                            round(x1, 2),
                            round(y1, 2),
                            round(x2, 2),
                            round(y2, 2),
                        ],
                    }
                )

        return detections