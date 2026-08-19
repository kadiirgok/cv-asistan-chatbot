# -*- coding: utf-8 -*-
"""
Modeli yükleyip basit bir sohbet testi yapan script.
Qwen2.5-1.5B modelini llama_cpp ile CPU üzerinde çalıştırır ve
"Merhaba, kendini tanıt" sorusunun cevabını + süresini ekrana yazar.
"""

import time                      # süre ölçümü için
from pathlib import Path         # dosya yollarını taşınabilir şekilde ele almak için

from llama_cpp import Llama      # llama.cpp'nin Python sarmalayıcısı

# Model dosyasının tam yolu: bu script src/ içinde, model ise ../models/ altında
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"


def main():
    print("Model yükleniyor (1.1 GB RAM'e alınacağı için ilk sefer biraz sürebilir)...")
    t0 = time.time()

    # Llama nesnesini oluştur: .gguf modelini belleğe yükler
    llm = Llama(
        model_path=str(MODEL_PATH),  # model dosyasının yolu
        n_ctx=4096,                  # bağlam penceresi: kaç token'lık sohbet hatırlanır
        n_threads=4,                 # kullanılacak CPU çekirdeği sayısı (i3 9. nesil = 4)
        verbose=False,               # yükleme sırasındaki uzun logları kapat
    )
    load_time = time.time() - t0
    print(f"Model yüklendi ({load_time:.1f} saniye).\n")

    # Sohbet tamamlama isteği gönder ve cevabın gelme süresini ölç
    print("Soru gönderiliyor: 'Merhaba, kendini tanıt'\n")
    t1 = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": "Merhaba, kendini tanıt"},
        ],
        max_tokens=256,    # üretilecek cevabın token üst sınırı
        temperature=0.7,   # yaratıcılık düzeyi (0 = tutarlı, 1 = daha yaratıcı)
    )
    elapsed = time.time() - t1

    # Cevabı yanıtın içinden çıkar
    answer = response["choices"][0]["message"]["content"]

    print("Cevap:")
    print(answer)
    print("\n" + "-" * 40)
    print(f"Cevap süresi: {elapsed:.2f} saniye")


if __name__ == "__main__":
    main()
