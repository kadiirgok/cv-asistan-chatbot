# -*- coding: utf-8 -*-
"""
Model + RAG ile terminalden sohbet eden script.

İş mantığını (retrieval + prompt + üretim) rag.py'deki generate_answer'a
bırakır; burada yalnızca model yükleme ve terminal döngüsü vardır.
"çıkış" yazılana kadar soru sormaya devam eder.
"""

import time

from llama_cpp import Llama

from rag import MODEL_PATH, generate_answer


def main():
    """Modeli yükler ve terminalden soru alarak RAG destekli sohbet döngüsü çalıştırır."""
    print("Model yükleniyor...")
    llm = Llama(
        model_path=MODEL_PATH,  # model dosyasının yolu (rag.MODEL_PATH ile seçilir)
        n_ctx=3072,                  # bağlam penceresi
        n_threads=4,                 # kullanılacak CPU çekirdeği sayısı
        verbose=False,               # yükleme loglarını kapat
    )
    print("Model hazır. Soru sorabilirsiniz; çıkmak için 'çıkış' yazın.\n")

    while True:
        # Kullanıcıdan soru al
        try:
            soru = input("Soru: ").strip()
        except EOFError:
            # Girdi kapanırsa (ör. pipe ile veri bitti) döngüyü sonlandır
            break

        if soru.lower() in ("çıkış", "cikis", "exit", "quit"):
            print("Görüşmek üzere!")
            break
        if not soru:  # boş girişi atla
            continue

        # RAG ile cevap üret ve süreyi ölç
        t0 = time.time()
        answer, kaynak, dogrulandi = generate_answer(llm, soru)
        elapsed = time.time() - t0

        print(f"\nYanıt ({elapsed:.2f} sn, kaynak: {kaynak}, doğrulandı: {dogrulandi}):")
        print(answer)
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
