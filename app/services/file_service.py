import os
import uuid


class FileService:

    def __init__(self):
        self.base_path = "data/inputs"
        os.makedirs(self.base_path, exist_ok=True)

    def save_file(self, filename: str, content: bytes) -> str:
        unique_name = f"{uuid.uuid4()}_{filename}"
        save_path = os.path.join(self.base_path, unique_name)

        with open(save_path, "wb") as f:
            f.write(content)

        return save_path