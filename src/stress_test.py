# -*- coding: utf-8 -*-
"""
RAG sistemini farklı soru tipleriyle değerlendiren stress test scripti.

Her soru için üç şeyi ekrana yazar:
1. Sorunun kendisi
2. retrieval_araci'nin döndürdüğü context (en yakın top_k chunk)
3. generate_answer'ın ürettiği model cevabı (+ süresi)

Amaç: yanlış chunk çekme, uydurma (halüsinasyon) ve yavaşlık gibi zayıf
noktaları gerçek veri eklemeden önce gözlemlemek.
"""

import time
from pathlib import Path

from llama_cpp import Llama

from rag import generate_answer, retrieval_araci

# Model dosyasının tam yolu
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"


def soru_test_et(llm, soru: str, baslik: str) -> None:
    """Bir soruyu retrieval + generate_answer ile çalıştırıp sonuçları yazar."""
    print("=" * 72)
    print(f"[{baslik}]")
    print(f"Soru: {soru}")

    # Retrieval'ın ne döndürdüğünü kısaca göster (detaylı mesafeler probe_distances.py'de)
    context, _source = retrieval_araci(soru)  # varsayılan top_k=4 + mesafe eşiği kullanılır
    print("\n--- Retrieval context (top_k=4, eşikli) ---")
    if context:
        print(" ".join(context.split()[:25]) + " ...")
    else:
        print("(boş)")

    # Model cevabını üret ve süresini ölç
    print("\n--- Model cevabı ---")
    t0 = time.time()
    cevap, kaynak, dogrulandi = generate_answer(llm, soru)
    sure = time.time() - t0
    print(cevap)
    print(f"\n(Kaynak: {kaynak}, doğrulandı: {dogrulandi}, süre: {sure:.2f} sn)")
    print("=" * 72 + "\n")


def main():
    """Modeli yükler ve dört farklı soru tipini art arda test eder."""
    print("Model yükleniyor...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        n_threads=4,
        verbose=False,
    )
    print("Model hazır. Stress test başlıyor...\n")

    # a) Cevabı tek bir chunk'ta olan basit soru
    soru_test_et(
        llm,
        "Deprem büyüklüğünü ölçmek için hangi ölçek kullanılır?",
        "a) Basit soru (tek chunk)",
    )

    # b) Cevabı farklı yerlerde olan, birleştirme gerektiren soru (tarih + ekonomi)
    soru_test_et(
        llm,
        "Bursa'nın tarihi önemini ve ekonomisinin temel sektörlerini anlat.",
        "b) Birleştirme gerektiren soru (çok chunk)",
    )

    # c) İndekste karşılığı olmayan, tamamen alakasız soru
    soru_test_et(
        llm,
        "Ay'a ilk insanlı iniş hangi yılda gerçekleşti?",
        "c) Alakasız soru (bilmiyorum beklenir)",
    )

    # d) Benzer ifadeli ama farklı dokümanlara ait iki soru (art arda)
    soru_test_et(
        llm,
        "Ege Bölgesi'nde yer alan şehir hakkında bilgi ver.",
        "d1) Benzer soru 1 (İzmir beklenir)",
    )
    soru_test_et(
        llm,
        "Marmara Bölgesi'nde yer alan şehir hakkında bilgi ver.",
        "d2) Benzer soru 2 (Bursa beklenir)",
    )

    print("Stress test tamamlandı.")


if __name__ == "__main__":
    main()
