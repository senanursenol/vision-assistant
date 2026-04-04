# Vision Assistant

AI-powered vision assistant system that analyzes images and videos by combining object detection (YOLO), OCR, and Vision-Language Models (VLM) with an intelligent routing mechanism.

---

## 🧠 Project Overview

This project aims to build a modular visual understanding system that can:

- Detect objects using YOLO
- Extract text using OCR
- Decide whether advanced reasoning is required
- Generate context-aware responses using Vision-Language Models (VLM)

The system is designed with a **score-based routing mechanism** to minimize unnecessary model usage and improve efficiency.

---

## ⚙️ System Architecture

Pipeline:

Input Image / Video  
↓  
Object Detection (YOLO)  
↓  
OCR (EasyOCR)  
↓  
Score-based Routing  
↙            ↘  
Fast Response     VLM (Qwen / MiniCPM)  
(Guidance)        (Deep Analysis)

---

## 🧩 Features

- Modular clean architecture
- YOLO-based object detection
- OCR text extraction
- Intelligent routing system (no unnecessary VLM calls)
- Extendable VLM integration (Qwen / MiniCPM)
- Ready for image and video processing

---

## 🧠 Routing Logic

The system calculates a **complexity score** based on:

- Number of detected objects
- Object diversity
- Detection confidence
- OCR presence

If the score exceeds a threshold, the request is forwarded to a VLM; otherwise, a lightweight response is generated.

---

## 📊 Model Evaluation

Two Vision-Language Models are evaluated:

- Qwen2.5-VL-3B-Instruct
- MiniCPM-V-4.5

### Evaluation Criteria

Each model is scored based on:

- Semantic Understanding (0–2)
- Task Relevance (0–2)
- Stability (0–2)
- Explanation Quality (0–2)

Total score per sample: **8 points**

Latency is also recorded for performance comparison.

---

## 🛠️ Tech Stack

- Python
- FastAPI
- YOLOv8 (Ultralytics)
- EasyOCR
- Transformers (for VLM integration)

---

## 📁 Project Structure

app/  
  api/  
  services/  
  core/  
  domain/  

data/  
scripts/  
tests/  

---

## 🚀 Getting Started

### 1. Create virtual environment

python -m venv .venv

### 2. Activate environment

.venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run API

uvicorn app.main:app --reload

---

## 🔮 Future Work

- Full VLM integration (Qwen / MiniCPM)
- Video processing pipeline
- Dynamic model routing
- Performance optimization with GPU
- Real-time inference

---

## 📌 Notes

This project is developed as a graduation thesis focusing on efficient multimodal AI systems and intelligent model usage.

---

## 👩‍💻 Author

Senanur Şenol
