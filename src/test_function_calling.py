# -*- coding: utf-8 -*-
"""
Function-calling destek testi (Adım 1).

Qwen2.5-1.5B-Instruct GGUF'un, llama-cpp-python üzerinden OpenAI-uyumlu
"tools" parametresiyle gerçekten bir tool_call döndürüp döndürmediğini doğrular.

Tek bir sahte araç ("hava_durumu_getir") tanımlanır, modele "İstanbul'da hava
nasıl?" diye sorulur ve dönen mesajda `tool_calls` olup olmadığı, hangi formatta
geldiği ekrana yazdırılır. Bu bilgi, Adım 2'deki agent katmanının kurulup
kurulmayacağına karar verir.
"""

import json
from pathlib import Path

from llama_cpp import Llama

# Model dosyasının tam yolu (src/ klasörünün bir üstündeki models/)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# OpenAI function-calling şemasına uygun TEK bir sahte araç tanımı
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "hava_durumu_getir",
            "description": "Verilen şehrin o anki hava durumunu döndürür.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sehir": {
                        "type": "string",
                        "description": "Hava durumu öğrenilmek istenen şehir adı.",
                    }
                },
                "required": ["sehir"],
            },
        },
    }
]

SORU = "İstanbul'da hava nasıl?"


def main():
    print("Model yükleniyor...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        n_threads=4,
        verbose=False,
    )
    print("Model hazır.\n")

    print("=" * 72)
    print(f"SORU: {SORU}")
    print("=" * 72 + "\n")

    # tools parametresiyle çağır (OpenAI-uyumlu)
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": SORU}],
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=256,
        temperature=0.0,
    )

    message = response["choices"][0]["message"]

    print("--- Modelden dönen ham 'message' ---")
    print(json.dumps(message, ensure_ascii=False, indent=2))
    print("=" * 72 + "\n")

    # --- Sonucu değerlendir ---
    tool_calls = message.get("tool_calls")
    content = message.get("content")

    print("--- SONUÇ ---")
    if tool_calls:
        print("BAŞARILI: Model bir tool_call döndürdü.")
        print(f"Araç sayısı: {len(tool_calls)}")
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            print(f"  [{i}] name      = {fn.get('name')}")
            print(f"      arguments = {fn.get('arguments')}")
            print(f"      id        = {tc.get('id')}")
            print(f"      type      = {tc.get('type')}")
        print("\n-> Qwen2.5-1.5B-Instruct GGUF, llama-cpp-python üzerinden")
        print("   function calling'i DESTEKLİYOR. Adım 2'ye geçilebilir.")
    else:
        print("BAŞARISIZ: Model tool_call DÖNDÜRMEDİ.")
        if content is not None:
            print(f"Bunun yerine düz metin cevabı verdi: {content!r}")
        else:
            print("Mesajda ne tool_calls ne de content var (boş çıktı).")
        print("\n-> Model, tool_call formatında yanıt üretmedi. Adım 2 kodlanmadan")
        print("   önce manuel prompt-tabanlı yönlendirme alternatifi değerlendirilmeli.")


if __name__ == "__main__":
    main()
