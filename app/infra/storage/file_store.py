from pathlib import Path
from uuid import uuid4

BASE_INPUT_DIR = Path("data/inputs")


def save_upload_file(filename: str, content: bytes) -> str:
    BASE_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    unique_filename = f"{uuid4()}_{filename}"
    file_path = BASE_INPUT_DIR / unique_filename

    with open(file_path, "wb") as f:
        f.write(content)

    return str(file_path)