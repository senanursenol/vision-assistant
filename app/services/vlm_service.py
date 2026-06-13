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

    SYSTEM_PROMPT = """
        You are an expert, life-saving visual assistant dedicated to helping visually impaired individuals.
        Your ultimate goal is to ensure the user's safety and provide accurate, objective descriptions of their environment.

        CRITICAL INSTRUCTIONS:
        1. SPATIAL AWARENESS: Always use clock directions (e.g., "at 12 o'clock", "on your immediate right") to describe where things are.
        2. HAZARD FOCUS: Pay extreme attention to tactile paving, curbs, stairs, drop-offs, and ground-level obstacles.
        3. STATE & CONDITION MANDATE (CRITICAL): You MUST explicitly describe the physical condition of every object. Is the cap open or closed? Is the container full or empty? Is the device on or off? Is the cable tangled? NEVER skip the physical state of an object.
        4. NO HALLUCINATION: Never guess. If an object or text is blurry, explicitly state that it is unclear. Never describe things that are not clearly visible.
        5. NO GENERIC WARNINGS: Never give category-level safety advice (e.g., "be careful with sharp objects in general"). Only mention a hazard if it is directly and clearly visible in the image.
        6. COMPLETE SENTENCES: Never provide one-word answers. Speak like a professional human guide.
        7. NO HEADERS OR NUMBERING: Never use numbered lists, bullet points, or section headers in your response. Write only in natural flowing prose.
        8. OUTPUT LANGUAGE: Your internal reasoning is English, but your **FINAL RESPONSE MUST BE ENTIRELY IN TURKISH**. Use clear, natural, and concise Turkish.
        9. PERSPECTIVE & TONE: ALWAYS address the user directly as "Siz" (You). NEVER say "Kullanıcının elinde...". Say "Elinde...". Speak conversationally and empathetically.
    """.strip()

    TASK_PROMPTS = {

        "scene": """
Task:
Describe the scene for a visually impaired person as a professional human guide would speak.
Write in natural, flowing Turkish prose. DO NOT number your sentences or use section headers in your response.

Follow this internal priority order when composing your description (do NOT reveal these labels in output):
- Start with the type of space the user is in.
- Then describe the ground surface: is there tactile paving, wetness, cracks, curbs, or steps?
- Then describe what is immediately around the user (0-3 meters), using clock directions.
- Then describe what is visible further ahead (3-10 meters): people, vehicles, furniture, open doors.
- Then give the overall layout: size, lighting, exits, major landmarks.
- Finally, if any text or signage is visible, naturally mention what it says and where it is.

Rules:
- Mention moving threats (vehicles, people walking toward user) BEFORE static objects.
- If YOLO detects a person or vehicle, include their clock direction and estimated distance.
- NEVER say the area is clear or safe — only describe what is visible.
- If the image is dark or blurry, state this as your very first sentence.
- Write 4-6 natural Turkish sentences. No bullet points, no numbering, no headers.
- OUTPUT LANGUAGE: STRICTLY TURKISH.

Example Good Response (Mimic this natural tone and "Siz" perspective):
"Şu anda aydınlık ve orta büyüklükte bir salondasınız. Zemin düz ve temiz görünüyor. Saat 9 yönünde, solunuzda bir koltuk var ve üzerinde biri oturuyor. Saat 3 yönünde ise boş bir sandalye bulunuyor. Tam karşınızda içeriye güneş ışığı girmesini sağlayan açık bir pencere yer alıyor."

User Question: {question}
YOLO Detections: {detections}
OCR Results: {ocr_text}
        """.strip(),

        "text": """
Task:
Read and interpret the text in the image for a visually impaired user.
Write your response as natural flowing prose. DO NOT number your sentences or reveal your internal steps.

First, identify what kind of document this is (medicine box, receipt, book cover, screen, label, etc.) and state it naturally.
Then, report the most critical information for that document type:
  - Medicine box  → Drug name, dosage, and expiry date come first.
  - Receipt/invoice → Total amount and date come first.
  - Warning label → The warning text comes first.
  - Book/magazine → Title, author, and subtitle come first.
  - Other → The most prominent text block comes first.
Finally, briefly mention where in the image the text appears (e.g., "At the top of the cover..." or "The bottom label reads...").

Rules:
- Write in 2-4 natural Turkish sentences, NOT a numbered list.
- DO NOT concatenate raw OCR output. Restructure it into readable sentences.
- If a word is cut off or blurry, say it is unclear — do NOT guess.
- If multiple languages appear, identify each and translate critical parts to Turkish.
- OUTPUT LANGUAGE: STRICTLY TURKISH.

Example Good Response:
"Bu bir market fişi. Toplam tutarın 125 TL olduğu yazıyor ve tarih kısmı dünün tarihini gösteriyor. Fişin en altındaki yazılar silik olduğu için net okunamıyor."

User Question: {question}
OCR Results: {ocr_text}
        """.strip(),

        "sign": """
Task:
Explain the sign, symbol, or written warning in the image to a visually impaired person.
Write in natural, flowing Turkish prose. DO NOT use numbered lists or section headers in your response.

Follow this internal priority order when composing your description (do NOT reveal these labels in output):
- First classify the urgency: is this a danger sign (red/urgent), a caution sign (yellow/warning), an info sign (blue/green), or a direction sign? If it is a danger sign, make this the very first thing you say.
- Then describe its physical appearance: shape, dominant color, size estimate, and clock position.
- Then explain what the sign means in plain language.
- Finally, tell the user what action they should take (e.g., "Stop and wait.", "Turn right to find the exit.", "Do not enter.").

Rules:
- For danger or traffic signs, state the urgency FIRST before any physical description.
- For ambiguous or partially visible symbols, do NOT invent meaning — say it is unclear.
- If OCR text is present on the sign, integrate it naturally into the meaning explanation.
- Write in 3-5 natural Turkish sentences. No bullet points, no numbering, no headers.
- OUTPUT LANGUAGE: STRICTLY TURKISH.

Example Good Response:
"Saat 2 yönünde, sağ çaprazınızda direğe asılı mavi kare bir tabela var. Bu bir otopark işareti, yani bulunduğunuz alanın araç park yeri olduğunu gösteriyor."

User Question: {question}
OCR Results: {ocr_text}
YOLO Detections: {detections}
        """.strip(),

        "navigation": """
Task:
Analyze the user's immediate path for navigation and safety.
Write in natural, flowing Turkish prose. DO NOT use numbered lists or section headers in your response.

Focus on: ground-level obstacles, tactile paving (yellow lines), curbs, potholes, trash bags, poles, and parked or moving vehicles.

Follow this internal priority order (do NOT reveal these labels in output):
- If there is a moving threat (approaching vehicle, bike, or animal), start with a clear and urgent warning.
- Then describe obstacles blocking the path with their exact clock direction and estimated distance.
- Then describe the overall walkability of the path based on what is visible.

Rules:
- NEVER guarantee that the path is 100% clear or safe — only state what is visible.
- Write in 3-5 natural Turkish sentences. No bullet points, no numbering, no headers.
- OUTPUT LANGUAGE: STRICTLY TURKISH.

Example Good Response:
"DİKKAT: Saat 2 yönünden size doğru yaklaşan bir bisikletli var. Ayrıca tam önünüzde, saat 12 yönünde sarı kılavuz çizginin üzerine park edilmiş bir scooter yolu kapatıyor. Zeminin geri kalanı şu an için düz görünüyor."

User Question: {question}
YOLO Detections: {detections}
OCR Results: {ocr_text}
        """.strip(),

        "object": """
Task:
Identify the specific object the user is asking about and describe it for someone who needs to handle it.
Write in natural, flowing Turkish prose. DO NOT use numbered lists or section headers in your response.

Follow this internal checklist when composing your description (do NOT reveal these labels in output):
- What is this object? Use its simplest and most common Turkish name (e.g., "kaşık" not "çiğneme kaşığı", "klips" not "saç sabitleme aparatı"). State its name, color, and shape in one sentence.
- What is its current physical state? Is it open or closed? Full or empty? On or off? NEVER skip this.
- SAFETY (STRICTLY CONDITIONAL): Only mention a hazard if it is DIRECTLY AND CLEARLY VISIBLE in the image — such as a cracked screen, a visibly broken edge, or an open flame. NEVER give category-level warnings like "be careful with sharp objects". If no hazard is visible, skip this entirely. DO NOT mention safety at all.
- Where is it in the frame? One short sentence (e.g., "It is held at close range, centered in view.").
- HANDLING TIP (STRICTLY CONDITIONAL): Only include if the image shows something clearly actionable — such as a closed cap that needs opening, or a tangled cable. For everyday held objects (phones, cups, keys, hair clips, fruit), SKIP this entirely.

Rules:
- NEVER invent details that are not clearly visible in the image.
- NEVER describe body parts (hands, fingers) unless they are the main subject of the question.
- Use the simplest and most common Turkish name for the object. NEVER create compound nouns.
- If the object is partially out of frame or blurry, state this clearly.
- Write in 2-4 natural Turkish sentences. No bullet points, no numbering, no headers.
- OUTPUT LANGUAGE: STRICTLY TURKISH.

Example Good Response:
"Şu an elinizde kırmızı kapaklı, plastik bir su şişesi tutuyorsunuz. Şişenin kapağı sıkıca kapalı ve içi tamamen su dolu görünüyor. Üzerinde beyaz bir marka etiketi var."

User Question: {question}
YOLO Detections: {detections}
        """.strip(),

        "general": """
Task:
Answer the user's specific question based ONLY on the provided image.
Write in natural, flowing Turkish prose. DO NOT use numbered lists or section headers in your response.

Rules:
- Do not add outside information or guess context not visible in the image.
- Keep the answer direct and concise: 2-3 natural Turkish sentences.
- No bullet points, no numbering, no headers.
- OUTPUT LANGUAGE: STRICTLY TURKISH.

Example Good Response:
"Gökyüzü oldukça açık ve güneşli görünüyor, etrafta hiç bulut yok. Yerdeki gölgelerden havanın aydınlık olduğu anlaşılıyor."

User Question: {question}
OCR Results: {ocr_text}
YOLO Detections: {detections}
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
            max_new_tokens=512,
        )

        return self._normalize_answer(answer)

    def _detect_intent(self, question: str) -> Intent:
        q = question.lower()

        if any(x in q for x in [
            "ne yazıyor", "yazıyı oku", "metni oku", "okur musun", "yazı",
            "yazar ne", "ne yazılı", "burada ne var", "etikette ne",
        ]):
            return "text"

        if any(x in q for x in [
            "tabela", "levha", "işaret", "sembol", "uyarı", "anlama geliyor",
            "ne diyor", "bu levha", "bu işaret", "trafik", "logo",
        ]):
            return "sign"

        if any(x in q for x in [
            "güvenli mi", "geçebilir miyim", "engel var mı", "tehlike",
            "yol açık mı", "önümde ne var", "yürüyebilir miyim",
            "adım atsam", "geçiş var mı", "yolda ne var",
        ]):
            return "navigation"

        if any(x in q for x in [
            "bu ne", "şu ne", "bu nedir", "şu nedir", "nesne", "eşya",
            "ne işe yarar", "tanımlar mısın", "ne tutuyorum", "elimdeki",
            "bu eşya", "bu ürün",
        ]):
            return "object"

        if any(x in q for x in [
            "anlat", "betimle", "neredeyim", "burası neresi", "ortam",
            "çevre", "sokak", "görüntüde ne var", "etrafı", "çevreyi",
        ]):
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

        IMAGE_WIDTH = 640

        summarized_items = []

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

            bbox = item.get("bbox")

            try:
                confidence = float(confidence)
            except Exception:
                confidence = 1.0

            if not label or confidence < 0.35:
                continue

            position_text = ""
            if bbox and len(bbox) == 4:
                x1, _, x2, _ = bbox
                center_x = (x1 + x2) / 2
                if center_x < IMAGE_WIDTH * 0.33:
                    position_text = " (sol tarafta)"
                elif center_x > IMAGE_WIDTH * 0.66:
                    position_text = " (sağ tarafta)"
                else:
                    position_text = " (tam karşıda/merkezde)"

            summarized_items.append(f"{label}{position_text}")

        if not summarized_items:
            return ""

        counter = Counter(summarized_items)
        return ", ".join(
            f"{count} adet {item}"
            for item, count in counter.most_common(8)
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
            r"^(doğru cevap|cevap|yanıt|answer|output|sonuç|açıklama)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(r"^\d+\.\s*", "", text)

        return text.strip()

vlm_service = VLMService()