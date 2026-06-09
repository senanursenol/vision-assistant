import re
import time
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

        replacements = {
            " visible": "",
            "visible": "",
            "Visible": "",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"\s+", " ", text).strip()

        unwanted_prefixes = [
            "Merhaba!",
            "Merhaba.",
            "Merhaba",
            "Ben bir asistanım.",
            "Ben bir sesli asistanım.",
        ]

        for prefix in unwanted_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        banned_phrases = [
            "bana bildir",
            "bana bildirin",
            "benimle paylaş",
            "size yardımcı olabilirim",
            "asistanıyım",
            "sesli asistan",
            "daha fazla detay",
            "bilgi verebilmem için",
            "size nasıl yardımcı",
            "bana daha fazla",
            "bu bilgileri paylaş",
        ]

        sentences = [s.strip() for s in text.split(".") if s.strip()]

        filtered_sentences = []
        for sentence in sentences:
            lower_sentence = sentence.lower()

            if any(phrase in lower_sentence for phrase in banned_phrases):
                continue

            filtered_sentences.append(sentence)

        filtered_sentences = filtered_sentences[:4]

        clean_text = ". ".join(filtered_sentences)

        if clean_text:
            clean_text += "."

        return clean_text.strip()

    def generate_description(self, image_path: str, detections: list, ocr_result: dict) -> str:
        self._load_model()

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

        return self._run_qwen(image_path=image_path, prompt=prompt, max_new_tokens=180)

    def answer_question(self, image_path: str, question: str, detections: list, ocr_result: dict) -> str:
        self._load_model()

        question = question.strip()
        lower_question = question.lower()
        ocr_text = ocr_result.get("text", "").strip() if ocr_result else ""

        safety_keywords = [
            "güvenli",
            "karşıya geç",
            "karşıdan karşıya",
            "geçebilir miyim",
            "yolu geç",
            "yoldan geç",
            "engel",
            "dikkat et",
            "tehlike",
            "araç",
            "trafik",
        ]

        is_safety_question = any(keyword in lower_question for keyword in safety_keywords)

        if is_safety_question:
            prompt = f"""
            You are an assistive visual question-answering assistant for a blind or visually impaired user.

            The user asks a safety-related question about the image:
            {question}

            Analyze the image carefully and answer based only on what is visible.

            Your answer should:
            - Be written only in Turkish.
            - Speak directly to the user.
            - Be specific to this image.
            - Explain what is visible in the image.
            - Mention relevant visual cues such as vehicles, pedestrians, road, crossing, traffic light, sidewalk, obstacles, or signs if they are visible.
            - Avoid giving a definite safety guarantee.
            - If safety cannot be fully determined from the image, say this clearly.
            - Still give practical and cautious guidance that helps the user move more safely.

            Do not:
            - Use bullet points, numbering, titles, or lists.
            - Give generic traffic advice unrelated to the image.
            - Ask the user for more information.
            - Invent exact distances, exact directions, or details that are not visible.
            - Say something is definitely safe.

            OCR text detected in the image:
            {ocr_text}

            Output:
            Write 2 or 3 natural Turkish sentences.
            The answer should sound like spoken guidance for the user.
            """
        else:
            prompt = f"""
            You are an assistive visual question-answering assistant for a blind or visually impaired user.

            The user asks this question about the image:
            {question}

            Answer the question based only on what is visible in the image.

            Your answer should:
            - Be written only in Turkish.
            - Speak directly to the user.
            - Be clear, useful, and practical.
            - Use readable text from OCR if it is relevant.
            - Clearly say if the answer is uncertain.

            Do not:
            - Use bullet points, numbering, titles, or lists.
            - Ask follow-up questions.
            - Invent details that are not visible.
            - Give a technical image caption.

            OCR text detected in the image:
            {ocr_text}

            Output:
            Write 2 or 3 natural Turkish sentences.
            The answer should directly answer the user's question.
            """

        return self._run_qwen(
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=180
    )
    
    def _run_qwen(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
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
                max_new_tokens=max_new_tokens,
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