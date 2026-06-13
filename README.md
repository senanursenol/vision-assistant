# 👁️ Vision Assistant

### Görme Engelliler İçin Yapay Zeka Destekli Gerçek Zamanlı Görsel Asistan

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

## 📌 Proje Hakkında

**Vision Assistant**, görme engelli ve az gören bireylerin çevrelerini daha bağımsız, güvenli ve erişilebilir şekilde algılayabilmelerini desteklemek amacıyla geliştirilmiş çok modlu bir yapay zeka destekli görsel asistan sistemidir.

Sistem, kullanıcının web arayüzü üzerinden sağladığı **kamera görüntüsünü** ve **sesli sorusunu** birlikte analiz ederek aşağıdaki çıktıları üretir:

- kısa ve anlaşılır sahne betimlemesi,
- nesne ve olası engel farkındalığı,
- görüntü üzerindeki metinlerin OCR ile okunması,
- görüntüye dayalı sesli soru-cevap,
- sesli komut alma ve sesli yanıt üretme.

Bu proje; **bilgisayarlı görü**, **görsel-dil modelleri**, **OCR**, **konuşma işleme** ve **modüler FastAPI backend mimarisini** modern bir web arayüzüyle birleştiren akademik bir prototip olarak geliştirilmiştir.

> Bu çalışma, sosyal fayda odaklı bir bitirme projesi kapsamında, gerçek zamanlı görsel anlama ve erişilebilir yapay zeka etkileşimi üzerine geliştirilmiştir.

---

## 🎯 Projenin Amacı

Görme engelli bireyler için geliştirilen birçok mevcut sistem sahne betimleme veya metin okuma gibi özellikler sunsa da çoğu kapalı kaynaklıdır, özelleştirilmesi zordur veya teknik performans açısından şeffaf değildir.

**Vision Assistant** projesinin amacı, aşağıdaki bileşenleri tek ve modüler bir sistem altında toplamaktır:

- çevresel görüntü analizi,
- nesne ve olası engel algılama,
- sahne içi metin okuma,
- görsel soru-cevap,
- sesli kullanıcı etkileşimi,
- API tabanlı genişletilebilir sistem mimarisi.

Proje yalnızca görsel açıklama üretmeyi değil, kullanıcının kamera görüntüsü hakkında **sesli soru sorabildiği** ve yanıtı yine **sesli olarak alabildiği** erişilebilir bir etkileşim akışı sunmayı hedefler.

---

## ✨ Özellikler

### 🧠 Görsel-Dil Modeli ile Sahne Anlama

Qwen / MiniCPM tabanlı görsel-dil modeli altyapısı ile kamera görüntüleri analiz edilir ve kullanıcıya Türkçe, kısa, somut ve bağlama uygun yanıtlar üretilir.

### 🔎 Nesne Tespiti

Ultralytics YOLO tabanlı nesne tespit altyapısı ile sahnedeki nesneler, çevresel öğeler ve potansiyel engeller algılanır.

### 📝 OCR ile Metin Okuma

EasyOCR entegrasyonu sayesinde tabela, ekran, belge, kağıt veya görüntü üzerindeki metinler tespit edilerek okunabilir hale getirilir.

### 🎙️ Sesli Soru Alma

Faster-Whisper kullanılarak kullanıcının sesli sorusu metne dönüştürülür.

### 🔊 Sesli Yanıt Üretme

Edge-TTS ile modelin ürettiği yanıt ses dosyasına dönüştürülür ve kullanıcıya sesli geri bildirim olarak sunulur.

### 🧭 Skor Tabanlı Yönlendirme

Kamera görüntüsü ve kullanıcının sesli isteğine göre sistem; OCR, nesne tespiti, sahne betimleme veya görsel soru-cevap akışlarından en uygun olanına yönlendirme yapabilir.

### 🌐 Modern Web Arayüzü

FastAPI üzerinden sunulan sade ve kullanıcı dostu web arayüzü ile kamera ve mikrofon tabanlı etkileşim desteklenir. Kullanıcı arayüzü, ekran veya klavye bağımlılığını azaltacak şekilde sesli etkileşim odaklı tasarlanmıştır.

### 🧩 Modüler Backend Mimarisi

Router, service, schema, config ve storage katmanlarına ayrılmış backend yapısı sayesinde sistem geliştirilebilir, sürdürülebilir ve genişletilebilir bir mimariye sahiptir.

---

## 🏗️ Sistem Mimarisi

```text
Kullanıcı
 │
 │  Kamera Görüntüsü + Sesli Soru
 ▼
Web Arayüzü
 │
 ▼
FastAPI Backend
 │
 ├── Health Router
 ├── Görsel Analiz Router
 ├── Görsel Soru-Cevap Router
 └── Sesli Soru-Cevap Router
 │
 ▼
Servis Katmanı
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
Çıktı
 ├── Sahne Betimlemesi
 ├── Algılanan Nesneler
 ├── Okunan Metin
 ├── Kullanıcı Sorusuna Yanıt
 └── Sesli Geri Bildirim
```

---

## 🛠️ Kullanılan Teknolojiler

| Kategori | Teknolojiler |
|---|---|
| Backend | FastAPI, Uvicorn |
| Programlama Dili | Python 3.10+ |
| Derin Öğrenme | PyTorch, Transformers, Hugging Face, Accelerate |
| Görsel-Dil Modeli | Qwen / MiniCPM tabanlı VLM altyapısı |
| Nesne Tespiti | Ultralytics YOLO |
| OCR | EasyOCR |
| Görüntü İşleme | OpenCV, Pillow, Scikit-image |
| Speech-to-Text | Faster-Whisper |
| Text-to-Speech | Edge-TTS |
| Veri ve Yardımcı Araçlar | NumPy, Polars, Pydantic, python-multipart |
| Arayüz | HTML, CSS, JavaScript, FastAPI Static Files |

---

## 📁 Proje Yapısı

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

## ⚙️ Kurulum

### 1. Repoyu klonlayın

```bash
git clone https://github.com/senanursenol/vision-assistant.git
cd vision-assistant
```

### 2. Sanal ortam oluşturun ve aktif edin

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

### 3. Bağımlılıkları yükleyin

Genel kurulum için:

```bash
pip install -r requirements.txt
```

GPU destekli RunPod benzeri ortamlarda, CUDA uyumlu PyTorch kurulumu sonrasında aşağıdaki dosya kullanılabilir:

```bash
pip install -r requirements_runpod.txt
```

> Görsel-dil modeli yüksek bellek gerektirebileceği için GPU destekli çalışma ortamı önerilir. Model boyutuna göre VRAM ihtiyacı değişebilir.

---

## ▶️ Uygulamayı Çalıştırma

Proje kök dizinindeyken FastAPI sunucusunu başlatın:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Geliştirme sırasında otomatik yenileme için:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Ardından tarayıcı üzerinden arayüze erişin:

```text
http://localhost:8000
```

FastAPI otomatik dokümantasyonu:

```text
http://localhost:8000/docs
```

Sistem durum kontrolü:

```text
http://localhost:8000/health
```

---

## 🔌 API Uç Noktaları

| Metot | Endpoint | Açıklama |
|---|---|---|
| `GET` | `/` | Ana web arayüzünü döndürür |
| `GET` | `/health` | API servisinin çalışıp çalışmadığını kontrol eder |
| `POST` | `/analyze/image` | Kamera görüntüsü veya görsel üzerinde genel analiz yapar |
| `POST` | `/ask/voice` | Kamera görüntüsü ve sesli soruyu alır, analiz eder ve sesli yanıt üretir |
| `POST` | `/ask/image` | API seviyesinde görsel ve metinsel soru ile görsel soru-cevap işlemi yapar |

> Ana kullanıcı akışı kamera görüntüsü ve sesli soru üzerinden çalışır. `/ask/image` endpoint'i API seviyesinde desteklenen ek bir görsel soru-cevap servisidir.

---

## 🧪 Örnek Kullanım Akışı

1. Kullanıcı web arayüzünü açar.
2. Sistem kamera ve mikrofon izinlerini ister.
3. Kullanıcı kamera görüntüsünü sistemle paylaşır.
4. Kullanıcı görüntü hakkında sesli soru sorar.
5. Sesli soru STT bileşeni ile metne dönüştürülür.
6. Sistem görüntüyü OCR, nesne tespiti ve görsel-dil modeli bileşenleriyle analiz eder.
7. Görsel-dil modeli kullanıcı sorusuna uygun yanıt üretir.
8. Üretilen yanıt TTS bileşeni ile sesli çıktıya dönüştürülür.
9. Kullanıcı yanıtı sesli olarak dinler.
10. Yanıt tamamlandıktan sonra kullanıcı yeni bir sesli soru sorabilir.

---

## 📊 Değerlendirme Kriterleri

Proje aşağıdaki ölçütler üzerinden değerlendirilebilir:

- uçtan uca gecikme süresi,
- OCR doğruluğu ve okunabilirlik başarımı,
- nesne tespit başarımı,
- görsel-dil modeli yanıtlarının bağlama uygunluğu,
- web arayüzünün kullanılabilirliği,
- sesli geri bildirimin anlaşılabilirliği,
- farklı ışık ve sahne karmaşıklığı koşullarında sistem kararlılığı.

---

## ⚠️ Sınırlılıklar

- Görsel-dil modeli CPU üzerinde yavaş çalışabilir.
- GPU bellek ihtiyacı seçilen model boyutuna göre değişebilir.
- Düşük ışık, bulanık görüntü ve karmaşık sahnelerde OCR doğruluğu azalabilir.
- Küçük veya kısmen görünen nesneler nesne tespit modeli tarafından kaçırılabilir.
- Sesli soru alma performansı ortam gürültüsünden etkilenebilir.
- Bu prototip, ek güvenlik testleri yapılmadan kritik navigasyon cihazı olarak kullanılmamalıdır.

---

## 🚀 Gelecek Geliştirmeler

- Gerçek zamanlı kamera akışı optimizasyonu.
- Sahne değişimine göre otomatik kare yakalama.
- Engel farkındalığı için mesafe tahmini.
- Mobil uyumlu arayüz geliştirmeleri.
- Hedef kullanıcılarla erişilebilirlik ve kullanılabilirlik testleri.
- Çok dilli kullanım desteği.
- Farklı GPU ortamlarında performans karşılaştırması.
- Docker ile taşınabilir dağıtım.

---

## 👩‍💻 Geliştirici

Bu proje, **senanursenol** tarafından bitirme projesi prototipi olarak geliştirilmiştir.

GitHub: `github.com/senanursenol`

---

## 📚 Akademik Bağlam

Bu çalışma; bilgisayarlı görü, doğal dil işleme ve konuşma teknolojilerinin erişilebilir yapay zeka sistemleri için bir araya getirilmesine odaklanmaktadır.

Projenin temel katkısı, görme engelli bireylerin çevresel farkındalığını artırmaya yönelik modüler, çok modlu, sesli etkileşim destekli ve web tabanlı bir görsel asistan prototipi sunmasıdır.

---

## 📄 Lisans

Bu proje için henüz açık kaynak lisansı belirtilmemiştir.  
Projeyi paylaşmadan veya yeniden kullanılabilir hale getirmeden önce MIT, Apache-2.0 veya GPL gibi uygun bir lisans dosyası eklenmesi önerilir.
