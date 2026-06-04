import sys
import time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python scripts/test_qwen.py <image_path>")
        return

    image_path = sys.argv[1]

    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Model yükleniyor...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

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
                    "text": (
                        "Yalnızca görüntüde gerçekten görünen bilgilere dayanarak cevap ver. "
                        "Görme engelli bir kullanıcıya yardımcı olacak şekilde Türkçe, kısa ve net açıkla. "
                        "Nesneleri tek tek listeleme; sahneyi ve kullanıcı için önemli riskleri belirt. "
                        "Tahmin yapma, görüntüde emin olmadığın şeyleri söyleme. "
                        "Cevap en fazla 3 cümle olsun."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    print("Inference başlıyor...")

    start = time.time()

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        repetition_penalty=1.05
    )

    end = time.time()

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print("\n--- Qwen Output ---")
    print(output_text[0])
    print(f"\nLatency: {round(end - start, 2)} seconds")


if __name__ == "__main__":
    main()
