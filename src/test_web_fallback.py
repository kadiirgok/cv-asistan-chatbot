# -*- coding: utf-8 -*-
"""
Web fallback + cache testi (kalite doğrulamalı).

Local DB'de kesinlikle olmayan güncel bir soruyu iki kez sorar:
  1. çalıştırma: web araması devreye girer (kaynak="web"), sonuç cache'lenir.
  2. çalıştırma: aynı soru artık cache'ten bulunur (kaynak="local"), daha hızlıdır.

Ayrıca web sonuçları ile modelin cevabı yan yana yazdırılır; böylece modelin
kaynaktaki bilgiye sadık kalıp kalmadığı (uydurup uydurmadığı) gözle görülür.
"""

import time
from pathlib import Path

from llama_cpp import Llama

from rag import generate_answer
from web_search import web_search_araci

# Model dosyasının tam yolu (src/ klasörünün bir üstündeki models/)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Local DB'de kesinlikle bulunmayan, güncel/spesifik bir soru
SORU = "2026 Kış Olimpiyatları hangi şehirde düzenlendi?"


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
    print("SORU:", SORU)
    print("=" * 72 + "\n")

    # --- Doğrulama: web sonuçlarını ve model cevabını yan yana göster ---
    print("--- WEB SONUÇLARI (modelin dayanacağı kaynak) ---")
    web_sonuclari = web_search_araci(SORU)
    print(web_sonuclari if web_sonuclari.strip() else "(web'den sonuç alınamadı)")
    print("=" * 72 + "\n")

    # --- 1. çalıştırma: web fallback beklenir ---
    print("[1. çalıştırma] generate_answer -> web fallback beklenir")
    t0 = time.time()
    cevap1, kaynak1, dogrulandi1 = generate_answer(llm, SORU)
    sure1 = time.time() - t0
    print(f"Kaynak      : {kaynak1}")
    print(f"Doğrulandı  : {dogrulandi1}")
    print(f"Süre        : {sure1:.2f} sn")
    print("MODELİN CEVABI:")
    print(cevap1)
    print("=" * 72 + "\n")

    # --- 2. çalıştırma: cache'ten (local) beklenir ---
    print("[2. çalıştırma] Aynı soru tekrar soruluyor (cache beklenir)...")
    t0 = time.time()
    cevap2, kaynak2, dogrulandi2 = generate_answer(llm, SORU)
    sure2 = time.time() - t0
    print(f"Kaynak      : {kaynak2}")
    print(f"Doğrulandı  : {dogrulandi2}")
    print(f"Süre        : {sure2:.2f} sn")
    print("MODELİN CEVABI:")
    print(cevap2)
    print("=" * 72 + "\n")

    # --- Sonuç + karşılaştırma ---
    print("--- Sonuç ---")
    print(f"1. çalıştırma kaynağı: {kaynak1}  (beklenen: web)")
    print(f"2. çalıştırma kaynağı: {kaynak2}  (beklenen: local)")
    print(f"1. süre: {sure1:.2f} sn  |  2. süre: {sure2:.2f} sn")
    if sure2 > 0:
        print(f"Hızlanma: 2. çalıştırma {sure1 / sure2:.1f}x daha hızlı")

    if kaynak1 == "web" and kaynak2 == "local":
        print("\nBAŞARILI: web fallback ve cache beklendiği gibi çalıştı.")
    elif kaynak1 == "none":
        print("\nWEB'E ULAŞILAMADI: ilk çalıştırma web'den sonuç alamadı (internet/tetikleme hatası).")
    else:
        print(f"\nBEKLENMEDİK SONUÇ: kaynak1={kaynak1}, kaynak2={kaynak2} (web -> local bekleniyordu).")


if __name__ == "__main__":
    main()
