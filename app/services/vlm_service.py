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
        # Çince/Japonca/Korece karakterleri temizle
        text = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", "", text)

        # Sık görülen İngilizce artıklarını temizle
        replacements = {
            " visible": "",
            "visible": "",
            "Visible": "",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Gereksiz boşlukları temizle
        text = re.sub(r"\s+", " ", text).strip()

        return text

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

        prompt = (
            "Sen görme engelli bir kullanıcıya çevresini anlatan bir sesli asistan gibi konuşuyorsun. "
            "Cevabın tamamen Türkçe olmalı ve kullanıcıyla doğrudan konuşur gibi olmalı. "
            "Resmi teknik olarak açıklama; kullanıcının bulunduğu ortamı anlamasına ve güvenli hareket etmesine yardım et. "
            "Sahneyi doğal bir şekilde anlat: şu an nerede olabiliriz, etrafta ne tür hareketlilik var, kullanıcı neye dikkat etmeli. "
            "Eğer görünüyorsa yol, araç, insan, trafik ışığı, kaldırım, geçit, tabela veya yürünebilir alan gibi bilgileri belirt. "
            "Görmediğin veya emin olmadığın şeyleri söyleme. "
            "Kesin mesafe verme; 300 metre, 5 metre gibi sayısal uzaklıklar uydurma. Bunun yerine 'yakında', 'ileride', 'sağ tarafta', 'sol tarafta' gibi ifadeler kullan. "
            "Maddeleme, numaralandırma ve başlık kullanma. "
            "Cevap 3 veya 4 doğal cümleden oluşsun. "
            "Örnek üslup: Şu an araçların geçtiği ve insanların yürüdüğü kalabalık bir yol kenarındasınız. "
            "İleride trafik ışığı görünüyor, bu yüzden karşıya geçmeden önce ışığı ve araç hareketlerini kontrol etmeniz önemli. "
            "Daha güvenli ilerlemek için kaldırım veya yaya yolu gibi ayrılmış bir alanı takip etmelisiniz."
        )

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