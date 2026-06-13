# 👁️ Vision Assistant

### AI-Powered Real-Time Visual Assistant for Visually Impaired Users

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688" />
  <img src="https://img.shields.io/badge/PyTorch-CUDA-red" />
  <img src="https://img.shields.io/badge/VLM-Qwen%20%2F%20MiniCPM-purple" />
  <img src="https://img.shields.io/badge/OCR-EasyOCR-orange" />
  <img src="https://img.shields.io/badge/STT-Faster--Whisper-lightgrey" />
  <img src="https://img.shields.io/badge/TTS-Edge--TTS-green" />
</p>

---

## 📌 Overview

**Vision Assistant** is a multimodal AI-powered visual assistant designed to help visually impaired and low-vision users understand their surroundings more independently, safely, and accessibly.

The system analyzes a **camera-based visual input** together with the user's **spoken question** and produces accessible outputs such as:

- short and understandable scene descriptions,
- object and possible obstacle awareness,
- OCR-based text reading from the image,
- voice-based visual question answering,
- spoken interaction through speech-to-text and text-to-speech.

This project combines **computer vision**, **vision-language models**, **OCR**, **speech processing**, and a **modular FastAPI backend** inside a clean web-based interface.

> This project was developed as an academic graduation project prototype focusing on real-time visual understanding and accessible AI interaction.

---

## 🎯 Project Motivation

Many existing assistive vision tools provide scene description or OCR capabilities, but they are often closed-source, limited in customizability, or difficult to evaluate from an engineering perspective.

**Vision Assistant** aims to provide a modular and extensible prototype that brings together:

- environmental image analysis,
- object and possible obstacle detection,
- scene text reading,
- visual question answering,
- voice-based user interaction,
- API-based extensible system design.

The goal is not only to generate visual descriptions, but also to create a practical interaction flow where the user can ask questions about a camera view and receive spoken responses.

---

## ✨ Features

### 🧠 Vision-Language Understanding

Uses a Vision-Language Model infrastructure such as **Qwen / MiniCPM** to understand image content and generate context-aware Turkish responses.

### 🔎 Object Detection

Integrates **Ultralytics YOLO-based detection** to identify objects, possible obstacles, and environmental elements in the scene.

### 📝 OCR Text Reading

Uses **EasyOCR** to detect and read visible text from signs, documents, screens, packages, and other image regions.

### 🎙️ Speech-to-Text

Uses **Faster-Whisper** to convert the user's spoken question into text.

### 🔊 Text-to-Speech

Uses **Edge-TTS** to convert the generated answer into an audio response.

### 🧭 Score-Based Routing

The system can route a camera image and user request toward the most relevant analysis flow, such as OCR, object detection, scene explanation, or VQA-style response generation.

### 🌐 Modern Web Interface

A FastAPI-served web interface supports camera and microphone-based interaction. The interface is designed to reduce keyboard and screen dependency by prioritizing voice-based interaction.

### 🧩 Modular Backend Architecture

The backend is divided into routers, services, schemas, configuration, and storage layers to keep the project maintainable, testable, and extensible.

---

## 🏗️ System Architecture

```text
User
 │
 │  Camera Snapshot + Voice Question
 ▼
Web Interface
 │
 ▼
FastAPI Backend
 │
 ├── Health Router
 ├── Image Analysis Router
 ├── Image Question Answering Router
 └── Voice Question Answering Router
 │
 ▼
Service Layer
 │
 ├── VLM Service
 ├── OCR Service
 ├── Detection Service
 ├── STT Service
 ├── TTS Service
 ├── Routing Service
 ├── Guidance Service
 └── File & Cleanup Services
 │
 ▼
Output
 ├── Scene Description
 ├── Detected Objects
 ├── Read Text
 ├── Answer to User Question
 └── Audio Feedback
```

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Backend | FastAPI, Uvicorn |
| Programming Language | Python 3.10+ |
| Deep Learning | PyTorch, Transformers, Hugging Face, Accelerate |
| Vision-Language Model | Qwen / MiniCPM-based VLM pipeline |
| Object Detection | Ultralytics YOLO |
| OCR | EasyOCR |
| Image Processing | OpenCV, Pillow, Scikit-image |
| Speech-to-Text | Faster-Whisper |
| Text-to-Speech | Edge-TTS |
| Data & Utilities | NumPy, Polars, Pydantic, python-multipart |
| Interface | HTML, CSS, JavaScript, FastAPI Static Files |

---

## 📁 Project Structure

```text
vision-assistant/
│
├── app/
│   ├── api/
│   │   └── routers/
│   │       ├── analyze_image.py
│   │       ├── ask_image.py
│   │       ├── ask_voice.py
│   │       └── health.py
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── domain/
│   │   └── schemas/
│   │
│   ├── infra/
│   │   └── storage/
│   │
│   ├── services/
│   │   ├── analyze_service.py
│   │   ├── ask_service.py
│   │   ├── ask_voice_service.py
│   │   ├── cleanup_service.py
│   │   ├── detection_service.py
│   │   ├── file_service.py
│   │   ├── guidance_service.py
│   │   ├── ocr_service.py
│   │   ├── routing_service.py
│   │   ├── stt_service.py
│   │   ├── tts_service.py
│   │   └── vlm_service.py
│   │
│   ├── static/
│   │   └── index.html
│   │
│   └── main.py
│
├── data/
│   └── outputs/
│
├── requirements.txt
├── requirements_runpod.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/senanursenol/vision-assistant.git
cd vision-assistant
```

### 2. Create and activate a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

For general installation:

```bash
pip install -r requirements.txt
```

For GPU-based RunPod-style environments, install the CUDA-compatible PyTorch build first, then install the RunPod-specific dependency file:

```bash
pip install -r requirements_runpod.txt
```

> GPU execution is recommended because the Vision-Language Model can require high VRAM depending on the selected model size.

---

## ▶️ Running the Application

Start the FastAPI server from the project root:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For development with automatic reload:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open the web interface:

```text
http://localhost:8000
```

FastAPI automatic documentation:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the main web interface |
| `GET` | `/health` | Checks whether the API is running |
| `POST` | `/analyze/image` | Performs general analysis on a camera image or image file |
| `POST` | `/ask/voice` | Takes a camera image and spoken question, then returns a spoken response |
| `POST` | `/ask/image` | API-level visual question answering using an image and a text question |

> The main user workflow is based on camera input and spoken questions. `/ask/image` is an additional API-level service for image-based text-question workflows.

---

## 🧪 Example Usage Flow

1. The user opens the web interface.
2. The system requests camera and microphone permissions.
3. The user shares a camera view with the system.
4. The user asks a spoken question about the visual scene.
5. The spoken question is converted into text by the STT component.
6. The system analyzes the image using OCR, object detection, and the Vision-Language Model.
7. The Vision-Language Model generates a context-aware response.
8. The generated response is converted into audio by the TTS component.
9. The user listens to the spoken answer.
10. After the answer is completed, the user can ask another spoken question.

---

## 📊 Evaluation Criteria

The project can be evaluated using the following criteria:

- end-to-end latency,
- OCR readability and accuracy,
- object detection quality,
- response relevance of the Vision-Language Model,
- usability of the web interface,
- clarity of the audio feedback,
- robustness under different lighting and scene complexity conditions.

---

## ⚠️ Limitations

- VLM inference can be slow on CPU-only systems.
- GPU memory requirements may vary depending on the selected model.
- OCR accuracy may decrease in low-light, blurry, or highly cluttered scenes.
- Object detection can miss small or partially visible objects.
- Speech-to-text performance may be affected by background noise.
- The current prototype should not be used as a safety-critical navigation device without further validation.

---

## 🚀 Future Improvements

- Real-time camera stream optimization.
- Scene-change-based frame triggering.
- Distance estimation for obstacle awareness.
- Mobile-friendly interface improvements.
- More detailed accessibility testing with target users.
- Multilingual support.
- Benchmarking on different GPU environments.
- Deployment with Docker.

---

## 👩‍💻 Developer

Developed by **senanursenol** as a graduation project prototype.

GitHub: `github.com/senanursenol`

---

## 📚 Academic Context

This project focuses on the integration of computer vision, natural language processing, and speech technologies for assistive AI systems.

Its main contribution is a modular, multimodal, voice-interaction-supported, web-based visual assistant prototype designed to improve environmental awareness for visually impaired users.

---

## 📄 License

No license has been specified yet.  
Before using or distributing this project, please add an appropriate open-source license such as MIT, Apache-2.0, or GPL depending on the intended usage.

