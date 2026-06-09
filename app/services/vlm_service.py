import re
import time
from collections import Counter

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class VLMService:
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(self):
        self.model = None
        self.processor = None
        self.last_latency = None

    def _load_model(self):
        if self.model is not None and self.processor is not None:
            return

        print("Qwen VLM yükleniyor...")

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=dtype,
            device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)

        print("Qwen VLM hazır.")

    def _format_detection_context(self, detections: list) -> str:
        if not detections:
            return "No reliable YOLO object detections."

        labels = []

        for detection in detections:
            label = detection.get("label", "")
            confidence = float(detection.get("confidence", 0.0))

            if label and confidence >= 0.35:
                labels.append(label)

        if not labels:
            return "No reliable YOLO object detections."

        counts = Counter(labels)

        return ", ".join(
            f"{label}: {count}"
            for label, count in counts.most_common(12)
        )

    def _classify_question_intent(self, question: str) -> str:
        q = question.lower().strip()

        sign_explanation_keywords = [
            "ne anlama geliyor",
            "ne anlama gelir",
            "ne demek",
            "anlamı ne",
            "işaret ne",
            "işaretin anlamı",
            "tabela ne anlama",
            "tabelanın anlamı",
            "levha ne anlama",
            "uyarı ne demek",
        ]

        text_reading_keywords = [
            "ne yazıyor",
            "yazıyı oku",
            "metni oku",
            "oku",
            "tabelada ne yazıyor",
            "levhada ne yazıyor",
            "etikette ne yazıyor",
            "menüde ne yazıyor",
            "ekranda ne yazıyor",
            "yazı var mı",
        ]

        safety_keywords = [
            "güvenli",
            "karşıya geç",
            "karşıdan karşıya",
            "geçebilir miyim",
            "geçebilir miyiz",
            "yolu geç",
            "yoldan geç",
            "engel var mı",
            "önümde engel",
            "önümde bir şey",
            "dikkat et",
            "tehlike",
            "yaklaşan",
            "çarp",
            "ilerleyebilir miyim",
            "yürüyebilir miyim",
            "araç geliyor mu",
            "trafik var mı",
        ]

        scene_description_keywords = [
            "anlat",
            "tarif et",
            "çevrem",
            "etraf",
            "ortam",
            "neredeyim",
            "nerede",
            "sokak",
            "cadde",
            "bulunduğum yer",
            "burada ne var",
            "ne görünüyor",
            "çevrede ne var",
            "etrafımda ne var",
            "bu görüntüde ne var",
            "bu ortamı anlat",
            "mekanı anlat",
            "manzarayı anlat",
        ]

        object_question_keywords = [
            "bu ne",
            "bu nedir",
            "şu ne",
            "şu nedir",
            "nesne ne",
            "ne işe yarıyor",
            "ne işe yarar",
            "soldaki",
            "sağdaki",
            "önümdeki",
            "arkadaki",
            "yakındaki",
        ]

        if any(keyword in q for keyword in sign_explanation_keywords):
            return "sign_explanation"

        if any(keyword in q for keyword in text_reading_keywords):
            return "text_reading"

        if any(keyword in q for keyword in safety_keywords):
            return "safety_navigation"

        if any(keyword in q for keyword in scene_description_keywords):
            return "scene_description"

        if any(keyword in q for keyword in object_question_keywords):
            return "object_question"

        return "general_vqa"

    def _clean_turkish_output(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        # Farklı alfabelerden karışan karakterleri temizle
        text = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04FF]", "", text)

        # Cevap etiketi, tırnak ve liste kalıntıları
        text = text.replace('"', "").replace("'", "")
        text = re.sub(r"^\s*(cevap|yanıt|answer|output)\s*[:：-]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(^|\s)[-*•]\s+", " ", text)
        text = re.sub(r"\b\d+[\).\s]+\s*", "", text)

        # Modelin kendini özne yapmasını azalt
        rewrite_patterns = [
            (r"\b[Bb]en\s+görüyorum\s*(ki)?", "Görüntüde"),
            (r"\b[Bb]en\s+görebiliyorum\s*(ki)?", "Görüntüde"),
            (r"\b[Gg]örüyorum\s*(ki)?", "Görüntüde"),
            (r"\b[Gg]örebiliyorum\s*(ki)?", "Görüntüde"),
            (r"\b[Bb]ana göre\b", "Görüntüye göre"),
        ]

        for pattern, replacement in rewrite_patterns:
            text = re.sub(pattern, replacement, text)

        unwanted_phrases = [
            "Merhaba!",
            "Merhaba.",
            "Merhaba",
            "Anladım,",
            "Anladım.",
            "Anladım",
            "Ben bir asistanım.",
            "Ben bir yapay zeka modeliyim.",
            "Ben bir sesli asistanım.",
            "Asistan olarak",
            "Bir asistan olarak",
            "Yapay zeka olarak",
            "visible",
            "Visible",
        ]

        for phrase in unwanted_phrases:
            text = text.replace(phrase, "")

        text = re.sub(r"\s+", " ", text).strip()

        banned_phrases = [
            "size nasıl yardımcı",
            "yardımcı olabilirim",
            "daha fazla bilgi verirseniz",
            "daha fazla detay verirseniz",
            "fotoğraf yükleyin",
            "görüntü yükleyin",
            "bu konuda yardımcı olamam",
            "sorunuzu daha açık",
            "ne hakkında bilgi almak",
            "bilgi vermek isterseniz",
            "bilgi vermek istersiniz",
            "nasıl gidebilirsiniz",
            "nasıl gidebilirsin",
        ]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        filtered_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            lower_sentence = sentence.lower()

            if any(phrase in lower_sentence for phrase in banned_phrases):
                continue

            # Açık birinci şahıs / sohbet kalıntıları
            if re.search(r"(^|\s)ben(\s|$)", lower_sentence):
                continue

            if (
                lower_sentence.startswith("bana ")
                or " benim " in lower_sentence
                or lower_sentence.startswith("benim ")
                or lower_sentence.startswith("anladım")
            ):
                continue

            if len(sentence.split()) < 2:
                continue

            filtered_sentences.append(sentence)

        filtered_sentences = filtered_sentences[:4]

        if not filtered_sentences:
            return (
                "Görüntü hakkında net ve güvenilir bir cevap üretilemedi. "
                "Kamerayı ilgili alana daha net yönelterek tekrar deneyebilirsiniz."
            )

        clean_text = " ".join(filtered_sentences).strip()

        if clean_text and clean_text[-1] not in ".!?":
            clean_text += "."

        return clean_text

    def generate_description(self, image_path: str, detections: list, ocr_result: dict) -> str:
        self._load_model()

        detection_context = self._format_detection_context(detections)
        ocr_text = ocr_result.get("text", "").strip() if ocr_result else ""

        prompt = f"""
You are a camera-based scene narrator for a blind or visually impaired user.

Task:
Describe the actual visible scene in the image. This is not a conversation and not a story.
Your output must help the user understand the surrounding environment.

Focus on:
- environment type: street, sidewalk, road, crosswalk, indoor area, shop, room, bus stop, etc.
- visible people, vehicles, roads, sidewalks, signs, traffic lights, doors, stairs, obstacles, tables, chairs
- approximate relative positions only if visible: ahead, left, right, background, near the ground
- crowding, movement, blocked paths, or open walking areas if visible
- readable text from OCR if useful
- practical guidance only after describing the visible scene

Rules:
- Answer only in Turkish.
- Do not greet the user.
- Do not introduce yourself.
- Do not role-play.
- Do not tell a story.
- Do not write as if you are inside the image.
- Do not describe any visible person as yourself.
- Do not use "ben", "bana", "benim", "ben görüyorum", "görüyorum", or "görebiliyorum".
- Do not use bullet points, numbering, titles, or lists.
- Do not invent exact distances or unseen details.
- Be concrete and image-specific.
- Avoid vague responses.

YOLO detected objects, secondary context and may be imperfect:
{detection_context}

OCR text detected in the image:
{ocr_text}

Output:
Write one natural Turkish paragraph with 3 or 4 sentences.
Sentence 1: describe the general environment.
Sentence 2: describe concrete visible elements.
Sentence 3: mention walking or safety-relevant details if visible.
Sentence 4: give brief practical guidance if useful.
"""

        return self._run_qwen(
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=260
        )

    def answer_question(self, image_path: str, question: str, detections: list, ocr_result: dict) -> str:
        self._load_model()

        question = question.strip()
        intent = self._classify_question_intent(question)

        ocr_text = ocr_result.get("text", "").strip() if ocr_result else ""
        detection_context = self._format_detection_context(detections)

        if intent == "scene_description":
            prompt = f"""
You are a camera-based scene narrator for a blind or visually impaired user.

The user is asking for an environment or scene description.
User request, used only to understand scope:
{question}

Task:
Describe the actual visible scene in the image. Do not answer as a chatbot.
Do not repeat the user's wording. Do not tell a story.

What to include:
- general environment type
- visible people, vehicles, buildings, road, sidewalk, shop, door, sign, table, chair, obstacle, or other important elements
- approximate relative positions only if visible
- crowding, movement, path openness, or possible obstacles if visible
- readable text from OCR if relevant

Rules:
- Answer only in Turkish.
- Do not greet the user.
- Do not introduce yourself.
- Do not role-play.
- Do not write as if you are inside the image.
- Do not describe any visible person as yourself.
- Do not say "ben", "bana", "benim", "ben görüyorum", "görüyorum", or "görebiliyorum".
- Do not use bullet points, numbering, titles, or lists.
- Do not invent exact distances or unseen details.
- Be concrete and image-specific.

YOLO detected objects, secondary context and may be imperfect:
{detection_context}

OCR text detected in the image:
{ocr_text}

Output:
Write 3 or 4 natural Turkish sentences.
First describe the visible environment, then give useful details for orientation.
"""

        elif intent == "text_reading":
            prompt = f"""
You are an OCR-focused visual assistant for a blind or visually impaired user.

The user asks:
{question}

Task:
Read visible text in the image. Use the OCR text if it is reliable, and also use the image if needed.
If the text is unclear or partially readable, say this clearly.

Rules:
- Answer only in Turkish.
- Do not greet the user.
- Do not introduce yourself.
- Do not role-play.
- Do not use "ben", "bana", "benim", "görüyorum", or "görebiliyorum".
- Do not invent text that is not visible.
- Do not use bullet points or numbering.
- If there is no readable text, say that no clear readable text is visible.
- If readable text exists, say it directly.

OCR text detected in the image:
{ocr_text}

YOLO detected objects, secondary context and may be imperfect:
{detection_context}

Output:
Write 1 to 3 natural Turkish sentences.
"""

        elif intent == "sign_explanation":
            prompt = f"""
You are a visual assistant for a blind or visually impaired user.

The user asks about the meaning of a sign, symbol, warning, or written notice:
{question}

Task:
Identify the visible sign or written element if possible. Explain what it likely means in daily life.
Use OCR text if relevant. If the sign or text is not clear, say this clearly.

Rules:
- Answer only in Turkish.
- Do not greet the user.
- Do not introduce yourself.
- Do not role-play.
- Do not use "ben", "bana", "benim", "görüyorum", or "görebiliyorum".
- Do not invent unseen signs or text.
- Do not use bullet points or numbering.
- Be practical and concise.
- If the sign affects navigation or safety, explain what the user should do cautiously.

OCR text detected in the image:
{ocr_text}

YOLO detected objects, secondary context and may be imperfect:
{detection_context}

Output:
Write 2 or 3 natural Turkish sentences.
"""

        elif intent == "safety_navigation":
            prompt = f"""
You are an assistive safety and navigation visual assistant for a blind or visually impaired user.

The user asks:
{question}

Task:
Answer using only visible information in the image. First describe relevant visible cues, then answer cautiously.
Do not give a definite safety guarantee.

Analyze if visible:
- road, sidewalk, crosswalk, curb, traffic light, sign
- vehicle, pedestrian, bicycle, motorcycle
- crowd density, blocked path, open path
- obstacles, stairs, doors, poles, barriers
- approximate relative positions only if visible

Rules:
- Answer only in Turkish.
- Speak directly to the user.
- Do not greet the user.
- Do not introduce yourself.
- Do not role-play.
- Do not use "ben", "bana", "benim", "görüyorum", or "görebiliyorum".
- Do not describe any visible person as yourself.
- Do not use bullet points, numbering, titles, or lists.
- Do not invent exact distances, exact directions, or unseen details.
- Do not say something is definitely safe.
- Do not give only a generic warning.
- If safety is uncertain, explain why using visible details.

YOLO detected objects, secondary context and may be imperfect:
{detection_context}

OCR text detected in the image:
{ocr_text}

Output:
Write 3 natural Turkish sentences.
Sentence 1: describe the relevant visible scene.
Sentence 2: answer the safety/navigation question cautiously.
Sentence 3: give practical guidance.
"""

        elif intent == "object_question":
            prompt = f"""
You are a visual question-answering assistant for a blind or visually impaired user.

The user asks about a visible object or part of the image:
{question}

Task:
Identify the relevant visible object if possible and explain it in a practical way.
If the target object is unclear, say what is visible and what is uncertain.

Rules:
- Answer only in Turkish.
- Speak directly to the user.
- Do not greet the user.
- Do not introduce yourself.
- Do not role-play.
- Do not use "ben", "bana", "benim", "görüyorum", or "görebiliyorum".
- Do not describe any visible person as yourself.
- Do not use bullet points or numbering.
- Do not invent details that are not visible.
- Be concrete and image-specific.

YOLO detected objects, secondary context and may be imperfect:
{detection_context}

OCR text detected in the image:
{ocr_text}

Output:
Write 2 or 3 natural Turkish sentences.
"""

        else:
            prompt = f"""
You are a visual question-answering assistant for a blind or visually impaired user.

The user asks:
{question}

Task:
Answer based only on visible information in the image.
Be concrete and image-specific. Do not answer as a general chatbot.

Rules:
- Answer only in Turkish.
- Speak directly to the user.
- Do not greet the user.
- Do not introduce yourself.
- Do not role-play.
- Do not write as if you are inside the image.
- Do not describe any visible person as yourself.
- Do not use "ben", "bana", "benim", "görüyorum", or "görebiliyorum".
- Do not use bullet points, numbering, titles, or lists.
- Do not ask follow-up questions.
- Do not invent details that are not visible.
- If the answer is uncertain, say it briefly but still describe what is visible.
- Use OCR text only if relevant.

YOLO detected objects, secondary context and may be imperfect:
{detection_context}

OCR text detected in the image:
{ocr_text}

Output:
Write 2 or 3 natural Turkish sentences.
Directly answer the user's question.
"""

        return self._run_qwen(
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=260
        )

    def _run_qwen(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
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

        start = time.time()

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3
            )

        end = time.time()
        self.last_latency = round(end - start, 2)

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return self._clean_turkish_output(output_text[0].strip())