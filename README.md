# 👁️ Vision Assistant (Görme Engelliler İçin Görsel Yapay Zeka Asistanı)

Vision Assistant, görme engelli ve az gören bireylerin günlük yaşamlarını kolaylaştırmak amacıyla geliştirilmiş, çok modlu (multimodal) yapay zeka destekli bir görsel asistan uygulamasıdır. Kullanıcıların yüklediği görselleri veya kamera görüntülerini analiz ederek sahne betimlemesi yapar, metinleri okur (OCR), nesneleri algılar ve kullanıcı ile **sesli** (Speech-to-Text & Text-to-Speech) etkileşime girer.

## ✨ Öne Çıkan Özellikler

- **Görsel Dil Modeli (VLM) Entegrasyonu:** Qwen2.5-VL tabanlı altyapı ile görselleri insan benzeri bir anlayışla analiz edip Türkçe ve duruma özel (somut, kısa) betimlemeler yapar.
- **Sesli Etkileşim:** `faster-whisper` ile kullanıcının sesli komutlarını yazıya döker (STT) ve `edge-tts` ile modelin cevaplarını doğal bir sesle kullanıcıya okur (TTS).
- **Optik Karakter Tanıma (OCR):** `EasyOCR` entegrasyonu sayesinde görüntüdeki tabela, ekran veya kağıt üzerindeki metinleri tespit edip okur.
- **Nesne Tespiti (Object Detection):** `Ultralytics` altyapısı ile çevredeki eşyaları, engelleri veya araçları hızlıca tespit eder.
- **Modern Web Arayüzü:** FastAPI tarafından sunulan, "Glassmorphism" ve modern "Soft-Gradient Mesh" tasarım diline sahip, kullanıcı dostu arayüz.
- **Modüler API Mimarisi:** Temiz kod (Clean Code) prensiplerine uygun olarak ayrılmış servisler (Yönlendirme, Ses, VLM, Analiz).

## 🛠️ Kullanılan Teknolojiler

- **Backend:** FastAPI, Uvicorn, Python 3.10+
- **Yapay Zeka & Derin Öğrenme:** PyTorch (CUDA 12.1), Transformers, HuggingFace
- **Görüntü İşleme & VLM:** Qwen-VL-Utils, OpenCV, Pillow, Scikit-image
- **Ses & Metin (Speech):** Faster-Whisper, Edge-TTS
- **Veri & Performans:** Polars, NumPy, Accelerate

## 🚀 Kurulum

Projeyi kendi bilgisayarında veya bir GPU sunucusunda çalıştırmak için aşağıdaki adımları izleyebilirsin.

### Ön Koşullar
- Python 3.10 veya üzeri
- NVIDIA GPU ve CUDA (CUDA 12.1 desteklenmektedir. VLM modelinin VRAM'e yüklenebilmesi için minimum 8GB+ VRAM önerilir.)

### 1. Depoyu Klonlayın
```bash
git clone [https://github.com/kullaniciadi/vision-assistant.git](https://github.com/kullaniciadi/vision-assistant.git)
cd vision-assistant

---

2. Sanal Ortam Oluşturun ve Aktif Edin
```bash
python -m venv venv
# Windows için:
venv\Scripts\activate
# Linux/MacOS için:
source venv/bin/activate

---

3. Bağımlılıkları Yükleyin
PyTorch GPU sürümü de dahil olmak üzere gerekli tüm paketleri kurmak için:
```bash
pip install -r requirements.txt

---

💻 Kullanım
Uygulamayı başlatmak için terminalde proje kök dizinindeyken aşağıdaki komutu çalıştırın:
```bash
python -m app.main

---

Sistem Başlatma Süreci:
API ayağa kalkarken VLM modeli otomatik olarak GPU VRAM'ine yüklenecektir. Konsolda şu mesajları göreceksiniz:

🚀 Sistem başlatılıyor: VLM Modeli GPU'ya yükleniyor (Lütfen bekleyin)...
✅ Model başarıyla VRAM'e yüklendi! Sistem isteklere hazır.

Sistem hazır olduğunda tarayıcınızdan http://localhost:8000 adresine giderek modern web arayüzüne erişebilirsiniz.

📡 API Uç Noktaları (Endpoints)
FastAPI otomatik dokümantasyonuna ulaşmak için uygulama çalışırken http://localhost:8000/docs adresini ziyaret edebilirsiniz.

GET /: Ana web arayüzünü (HTML) döndürür.

GET /api/health: Sistem durumunu ve modelin yüklenip yüklenmediğini kontrol eder.

POST /api/analyze-image: Verilen görseli genel olarak analiz eder (Sahne, metin, nesne).

POST /api/ask-image: Görsel ile birlikte kullanıcıdan gelen spesifik yazılı soruyu yanıtlar.

POST /api/ask-voice: Kullanıcıdan alınan ses kaydını (STT) metne çevirir, VLM'e sorar ve yanıtı ses dosyası (TTS) olarak döndürür.

📂 Proje Yapısı
vision-assistant/
├── app/
│   ├── api/
│   │   └── routers/ (analyze_image.py, ask_image.py, ask_voice.py, vb.)
│   ├── core/ (config.py - Ortam ve ayar değişkenleri)
│   ├── domain/ (Pydantic şemaları)
│   ├── infra/ (Depolama, dosya yönetimi vb.)
│   ├── services/ (vlm_service, stt_service, tts_service, ocr_service vb.)
│   ├── static/ (Web arayüzü dosyaları - index.html)
│   └── main.py (FastAPI uygulaması başlangıç noktası)
├── data/
│   └── outputs/ (Üretilen ses dosyaları veya geçici analiz çıktıları)
├── requirements.txt
└── README.md


