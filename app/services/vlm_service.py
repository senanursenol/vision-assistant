import re
from collections import Counter
from typing import Literal

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

Intent = Literal[
    "scene",
    "text",
    "sign",
    "navigation",
    "object",
    "general",
]

class VLMService:
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    # GÜNCELLENEN KISIM: Görme engelliler için ekstra hassasiyet eklendi
    SYSTEM_PROMPT = """
Sen görme engelli bireyler için hayat kurtarıcı profesyonel bir görsel asistansın.
Amacın kullanıcının güvenliğini sağlamak ve çevreyi ona en doğru şekilde betimlemektir.
ÖZELLİKLE DİKKAT ET: Görüntüdeki sarı hissedilebilir yürüme yüzeylerine (kılavuz iz), kaldırımlara, çukurlara ve yoldaki engellere (çöp, direk, araç, tabela vb.) odaklan.
Yanıtlarını her zaman Türkçe, tek paragraf ve net ver. Halüsinasyon yapma, görüntüde olmayan hiçbir şeyi uydurma.
""".strip()

    TASK_PROMPTS = {
        "scene": """
Kullanıcı sorusu:
{question}

Görev:
Görüntüdeki sahneyi nesnel şekilde anlat.

Odak:
Genel ortam, yol, bina, araç, insan, ışık, kapı, merdiven, tabela ve belirgin nesneler.

Kural:
Sadece görünür olanı söyle.
Yol güvenli, geçiş açık, insan yok, başka yol yok gibi kesin çıkarımlar yapma.
""".strip(),

        "text": """
Kullanıcı sorusu:
{question}

Görev:
Görüntüdeki okunabilir yazıyı söyle veya özetle.

OCR sonucu:
{ocr_text}

Kural:
Yazı net değilse bunu söyle.
Eksik kelime uydurma.
""".strip(),

        "sign": """
Kullanıcı sorusu:
{question}

Görev:
Görüntüdeki tabela, işaret, sembol veya yazılı uyarıyı açıkla.

OCR sonucu:
{ocr_text}

Kural:
İşaret veya yazı net değilse kesin anlam verme.
""".strip(),

        # GÜNCELLENEN KISIM: Sarı çizgi ve engel hassasiyeti artırıldı
        "navigation": """
Kullanıcı sorusu:
{question}

Görev:
Görüntüyü bir görme engellinin gözünden incele. Yerdeki engelleri, özellikle sarı hissedilebilir yürüme yüzeyinin (kılavuz izin) üzerindeki tehlikeleri (çöp, çöp poşeti, araç, eşya) ve yolu kapatan unsurları detaylıca söyle.

Nesne tespiti (YOLO):
{detections}

OCR sonucu:
{ocr_text}

Kural:
Asla "Yol tamamen açık ve güvenli" gibi kesin bir garanti verme.
Sadece görüntüde açıkça görünen tehlikelerden bahset.
""".strip(),

        "object": """
Kullanıcı sorusu:
{question}

Görev:
Görüntüde sorulan nesneyi tanımla.

Nesne tespiti:
{detections}

Kural:
Nesne net değilse net olmadığını söyle.
""".strip(),

        "general": """
Kullanıcı sorusu:
{question}

Görev:
Soruyu sadece görüntüye göre cevapla.

OCR sonucu:
{ocr_text}

Nesne tespiti:
{detections}

Kural:
Görüntüde olmayan ayrıntı ekleme.
""".strip(),
    }

    def __init__(self):
        self.model = None
        self.processor = None

    def _load_model(self):
        if self.model is not None and self.processor is not None:
            return

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)

    def generate_description(self, image_path: str, detections: list, ocr_result: dict) -> str:
        return self.answer_question(
            image_path=image_path,
            question="Bu görüntüdeki ortamı anlat.",
            detections=detections,
            ocr_result=ocr_result,
        )

    def answer_question(
        self,
        image_path: str,
        question: str,
        detections: list | None = None,
        ocr_result: dict | None = None,
    ) -> str:
        self._load_model()

        intent = self._detect_intent(question)
        prompt = self._build_prompt(
            intent=intent,
            question=question,
            detections=detections or [],
            ocr_result=ocr_result or {},
        )

        answer = self._generate(
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=180,
        )

        return self._normalize_answer(answer)

    def _detect_intent(self, question: str) -> Intent:
        q = question.lower()

        if any(x in q for x in ["ne yazıyor", "yazıyı oku", "metni oku", "okur musun", "yazı"]):
            return "text"

        if any(x in q for x in ["tabela", "levha", "işaret", "sembol", "uyarı", "anlama geliyor"]):
            return "sign"

        if any(x in q for x in ["güvenli mi", "geçebilir miyim", "engel var mı", "tehlike", "yol açık mı", "önümde ne var"]):
            return "navigation"

        if any(x in q for x in ["bu ne", "şu ne", "nesne", "eşya", "ne işe yarar"]):
            return "object"

        if any(x in q for x in ["anlat", "betimle", "neredeyim", "burası neresi", "ortam", "çevre", "sokak", "görüntüde ne var"]):
            return "scene"

        return "general"

    def _build_prompt(
        self,
        intent: Intent,
        question: str,
        detections: list,
        ocr_result: dict,
    ) -> str:
        ocr_text = self._extract_ocr_text(ocr_result)
        detection_text = self._summarize_detections(detections)

        template = self.TASK_PROMPTS[intent]

        return template.format(
            question=question.strip(),
            ocr_text=ocr_text if ocr_text else "Güvenilir OCR metni yok.",
            detections=detection_text if detection_text else "Belirgin nesne tespiti yok.",
        )

    def _generate(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output.strip()

    def _extract_ocr_text(self, ocr_result: dict) -> str:
        if not ocr_result:
            return ""

        text = ocr_result.get("text", "")

        if isinstance(text, list):
            return " ".join(str(item) for item in text).strip()

        return str(text).strip()

    def _summarize_detections(self, detections: list) -> str:
        if not detections:
            return ""

        counter = Counter()

        for item in detections:
            if not isinstance(item, dict):
                continue

            label = (
                item.get("label")
                or item.get("name")
                or item.get("class_name")
                or item.get("class")
            )

            confidence = (
                item.get("confidence")
                or item.get("score")
                or item.get("conf")
                or 1.0
            )

            try:
                confidence = float(confidence)
            except Exception:
                confidence = 1.0

            if label and confidence >= 0.35:
                counter[str(label)] += 1

        if not counter:
            return ""

        return ", ".join(
            f"{count} adet {label}"
            for label, count in counter.most_common(8)
        )

    def _normalize_answer(self, text: str) -> str:
        if not text:
            return "Görüntü hakkında net bir cevap üretilemedi."

        text = text.strip()

        text = text.replace("。", ".")
        text = text.replace("，", ",")
        text = text.replace("？", "?")
        text = text.replace("！", "!")

        text = re.sub(r"[\u4e00-\u9fff]+", "", text)
        text = re.sub(r"[\r\n]+", " ", text)
        text = re.sub(r"\s+", " ", text)

        text = re.sub(
            r"^(cevap|yanıt|answer|output)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        return text.strip()

vlm_service = VLMService()