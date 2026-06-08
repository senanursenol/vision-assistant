import time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re

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

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)

        print("Qwen VLM hazır.")

    def _clean_turkish_output(self, text: str) -> str:
        # Çince/Japonca/Korece + Kiril karakterlerini temizle
        text = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04FF]", "", text)

        # Gereksiz tırnakları temizle
        text = text.replace('"', "").replace("'", "")

        # Sık görülen yabancı artıklar
        replacements = {
            " visible": "",
            "visible": "",
            "Visible": "",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Gereksiz boşlukları temizle
        text = re.sub(r"\s+", " ", text).strip()

        # İstenmeyen başlangıçları temizle
        unwanted_prefixes = [
            "Merhaba!",
            "Merhaba.",
            "Merhaba",
            "Ben bir asistanım.",
            "Ben bir sesli asistanım.",
            "Kendiniz bir sesi olarak,",
        ]

        for prefix in unwanted_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # Chatbot gibi konuşan cümleleri at
        banned_phrases = [
            "bana bildir",
            "bana bildirin",
            "benimle paylaş",
            "size yardımcı olabilirim",
            "asistanıyım",
            "sesli asistan",
        ]

        sentences = [s.strip() for s in text.split(".") if s.strip()]

        filtered_sentences = []
        for sentence in sentences:
            lower_sentence = sentence.lower()

            if any(phrase in lower_sentence for phrase in banned_phrases):
                continue

            filtered_sentences.append(sentence)

        # En fazla 3 cümle bırak
        filtered_sentences = filtered_sentences[:3]

        clean_text = ". ".join(filtered_sentences)

        if clean_text:
            clean_text += "."

        return clean_text.strip()

    def generate_description(self, image_path: str, detections: list, ocr_result: dict) -> str:
        self._load_model()

        label_tr_map = {
            "person": "kişi",
            "car": "araç",
            "traffic light": "trafik ışığı",
            "bus": "otobüs",
            "truck": "kamyon",
            "motorcycle": "motosiklet",
            "bicycle": "bisiklet",
            "chair": "sandalye",
            "bench": "bank",
            "cup": "bardak",
            "bowl": "kase",
            "dog": "köpek",
            "cat": "kedi",
        }

        detected_labels = sorted(list(set([d["label"] for d in detections])))
        detected_labels_tr = [
            label_tr_map.get(label, label)
            for label in detected_labels
        ]

        ocr_text = ocr_result.get("text", "").strip() if ocr_result else ""

        helper_context = ""

        if detected_labels_tr:
            helper_context += f"Algılanan nesneler: {', '.join(detected_labels_tr)}. "


        detected_labels = sorted(list(set([d["label"] for d in detections])))
        ocr_text = ocr_result.get("text", "").strip() if ocr_result else ""

        helper_context = ""

        if detected_labels:
            helper_context += f"YOLO tarafından algılanan nesneler: {', '.join(detected_labels)}. "

        if ocr_text:
            helper_context += f"OCR tarafından okunan metin: {ocr_text}. "

        prompt = """
        You are an assistive visual navigation assistant for a blind or visually impaired user.

        Your task:
        Analyze the image and describe the surrounding environment in a way that helps the user understand where they are and move more safely.

        Focus on:
        - the type of environment
        - nearby people, vehicles, roads, sidewalks, crossings, signs, traffic lights, or walkable areas if visible
        - possible safety risks
        - practical guidance for safe movement

        Response style:
        - Speak directly to the user, as if you are walking beside them.
        - Do not describe yourself.
        - Do not greet the user.
        - Do not ask questions.
        - Do not use bullet points, titles, numbering, or lists.
        - Do not simply list objects.
        - Do not invent exact distances, directions, or details that are not visible.
        - Use natural, calm, and clear language.

        Output requirements:
        - Write only in Turkish.
        - Write one short paragraph.
        - Use 3 to 4 natural sentences.
        - The answer should sound like spoken guidance, not a technical image caption.
        """
         
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        start = time.time()

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                repetition_penalty=1.1,
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