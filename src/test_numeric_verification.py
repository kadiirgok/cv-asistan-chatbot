# -*- coding: utf-8 -*-
"""
Sayısal doğrulamanın testi.

İki bölüm:
  1. dogrula_sayisal_tutarlilik() için birim testler (6 senaryo, model gerektirmez).
  2. generate_answer() ile gerçek bir web sorgusu (uçtan uca) — `dogrulandi`
     alanının ve nihai cevabın ne döndüğünü gösterir.
"""

import time
from pathlib import Path

from llama_cpp import Llama

from rag import dogrula_eksik_deger, dogrula_sayisal_tutarlilik, generate_answer
from web_search import web_search_araci

# Model dosyasının tam yolu (src/ klasörünün bir üstündeki models/)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# (açıklama, cevap, kaynak_metin, beklenen_sonuç) dörtlüleri
BIRIM_TESTLER = [
    ("doğru sayı (tam eşleşme)",
     "İstanbul'un fethi 1453 yılında oldu.",
     "İstanbul 1453'te fethedildi.",
     True),
    ("uydurma sayı (kaynakta yok)",
     "Dolar kuru 1.47 TL.",
     "Dolar kuru 35.24 TL seviyesinde.",
     False),
    ("sayısız cevap",
     "Bu konuda net bir bilgi yok.",
     "Herhangi bir kaynak metni.",
     True),
    ("yüzde formatı (%20)",
     "Üründe %20 indirim var.",
     "İndirim oranı %20.",
     True),
    ("yuvarlama toleransı (35.2 vs 35.24)",
     "Kur 35.2 TL.",
     "Kur 35.24 TL.",
     True),
    ("cevapta sayı var, kaynakta hiç yok",
     "Sıcaklık 30 derece.",
     "Bugün hava güneşli.",
     False),
]

# (açıklama, cevap, kaynak_metin, beklenen_sonuç) dörtlüleri — ters yön: eksik değer.
# dogrula_eksik_deger kaynaktaki sayıların cevapta eksik kalıp kalmadığına bakar.
EKSIK_DEGER_TESTLER = [
    ("kaynakta 3 sayı, cevapta hepsi var",
     "Sonuçlar 0.85, 1.0 ve 1.5 olarak ölçüldü.",
     "Ölçüm değerleri 0.85, 1.0 ve 1.5 çıktı.",
     True),
    ("kaynakta 3 sayı, cevapta sadece 1'i var (eksik liste)",
     "Sonuç 1.0 çıktı.",
     "Ölçüm değerleri 0.85, 1.0 ve 1.5 çıktı.",
     False),
    ("kaynakta 1 sayı, cevapta o sayı var",
     "Sonuç 0.85 çıktı.",
     "Ölçüm değeri 0.85 çıktı.",
     True),
    ("kaynakta dağınık sayılar (27, 400, 60), cevapta sadece biri",
     "Arşivde 27 makale var.",
     "Arşivde 27 makale var, chunk boyutu 400 ve overlap 60.",
     True),
    ("kaynakta net liste (@1=0.85, @3=1.0, @5=1.0), cevapta sadece 1.0",
     "Hit-rate sonucu 1.0.",
     "Hit-rate sonuçları @1=0.85, @3=1.0, @5=1.0.",
     False),
]


def test_birim() -> bool:
    """dogrula_sayisal_tutarlilik birim testlerini çalıştırır; tümü geçtiyse True."""
    print("=" * 72)
    print("BÖLÜM 1 — dogrula_sayisal_tutarlilik() birim testleri")
    print("=" * 72)

    tum_ok = True
    for aciklama, cevap, kaynak, beklenen in BIRIM_TESTLER:
        sonuc = dogrula_sayisal_tutarlilik(cevap, kaynak)
        ok = sonuc == beklenen
        tum_ok = tum_ok and ok
        print(f"[{'GEÇTİ' if ok else 'HATA'}] {aciklama}")
        print(f"       cevap   = {cevap!r}")
        print(f"       kaynak  = {kaynak!r}")
        print(f"       beklenen= {beklenen}, alınan = {sonuc}")

    print("-" * 72)
    print("Birim testler:", "TÜMÜ GEÇTİ" if tum_ok else "BAZILARI BAŞARISIZ")
    return tum_ok


def test_eksik_deger() -> bool:
    """dogrula_eksik_deger birim testlerini çalıştırır; tümü geçtiyse True."""
    print("=" * 72)
    print("BÖLÜM 1b — dogrula_eksik_deger() birim testleri")
    print("=" * 72)

    tum_ok = True
    for aciklama, cevap, kaynak, beklenen in EKSIK_DEGER_TESTLER:
        sonuc = dogrula_eksik_deger(cevap, kaynak)
        ok = sonuc == beklenen
        tum_ok = tum_ok and ok
        print(f"[{'GEÇTİ' if ok else 'HATA'}] {aciklama}")
        print(f"       cevap   = {cevap!r}")
        print(f"       kaynak  = {kaynak!r}")
        print(f"       beklenen= {beklenen}, alınan = {sonuc}")

    print("-" * 72)
    print("Eksik değer testleri:", "TÜMÜ GEÇTİ" if tum_ok else "BAZILARI BAŞARISIZ")
    return tum_ok


def test_uc_uca() -> None:
    """generate_answer'ı gerçek bir güncel/sayısal web sorgusuyla çalıştırır."""
    print("\n" + "=" * 72)
    print("BÖLÜM 2 — generate_answer() uçtan uca (web sorgusu)")
    print("=" * 72)

    # Local DB'de olmayan, güncel ve sayısal bir soru (web fallback beklenir).
    soru = "Güncel altın fiyatı kaç TL?"

    print("Model yükleniyor...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        n_threads=4,
        verbose=False,
    )
    print("Model hazır.\n")

    print(f"SORU: {soru}\n")

    # Modelin dayanacağı web sonuçlarını ayrıca göster (şeffaflık için).
    print("--- Web sonuçları (modelin dayanacağı kaynak) ---")
    web_sonuc = web_search_araci(soru)
    print(web_sonuc if web_sonuc.strip() else "(web'den sonuç alınamadı)")
    print("=" * 72 + "\n")

    t0 = time.time()
    cevap, kaynak, dogrulandi = generate_answer(llm, soru)
    sure = time.time() - t0

    print("--- generate_answer sonucu ---")
    print(f"Kaynak     : {kaynak}")
    print(f"Doğrulandı : {dogrulandi}")
    print(f"Süre       : {sure:.2f} sn")
    print("Nihai cevap:")
    print(cevap)


def main() -> None:
    ok = test_birim()
    ok = test_eksik_deger() and ok
    test_uc_uca()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
