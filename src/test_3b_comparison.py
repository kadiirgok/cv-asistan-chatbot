# -*- coding: utf-8 -*-
"""
1.5B, 3B ve 7B modellerini aynı 4 test sorusu üzerinde karşılaştırır.

Aynı prompt/parametreler (max_tokens=150, temperature=0.1, repeat_penalty=1.3,
n_ctx=3072, n_threads=4) ile üç modeli sırayla çalıştırır, cevapları ve süreleri
kaydeder, ÜÇLÜ bir tabloda (1.5B / 3B / 7B yan yana) karşılaştırır. Bellek
kullanımını da mümkünse (psutil ile) ölçer. Production'ı (api.py) DEĞİŞTİRMEZ;
sadece ölçüm içindir.

Sonuçlar ayrıca src/test_3b_sonuclari.json dosyasına yazılır (rapor için).
"""

import gc
import json
import time
from pathlib import Path

from llama_cpp import Llama

from rag import (
    MODEL_PATH_15B,
    MODEL_PATH_3B,
    MODEL_PATH_7B,
    generate_answer,
    load_embedding_model,
)

SORULAR = [
    "BilgiTR projesinde hit-rate sonucu nedir?",
    "YorumTR projesinde hangi yöntem daha iyi sonuç verdi?",
    "HizmetGelsin'in backend'i hangi teknolojilerle yazıldı?",
    "Telco churn projesinde recall skoru nasıl iyileşti?",
]

# Soru -> "doğru olguyu içeriyor mu" kontrol anahtarları.
# Q1 (indeks 0): hit-rate değerleri (0.85 ve 1.0). Q4 (indeks 3): veri sızıntısı
# düzeltmesi (IQRClipper + sızıntı mekanizması). Diğerleri serbest cevap.
KONTROL = {
    0: ["0.85", "1.0"],            # Q1: hit-rate değerleri
    3: ["IQRClipper", "sızıntı"],  # Q4: veri sızıntısı düzeltmesi / IQRClipper
}

N_CTX = 3072
N_THREADS = 4

# Sıra ve etiketler: 1.5B -> 3B -> 7B
MODELLER = [
    ("1.5B", MODEL_PATH_15B),
    ("3B", MODEL_PATH_3B),
    ("7B", MODEL_PATH_7B),
]

JSON_CIKTI = Path(__file__).resolve().parent / "test_3b_sonuclari.json"


def _rss_mb():
    """Sürecin RSS (resident) bellek kullanımını MB cinsinden döndürür (psutil yoksa None)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _vms_mb():
    """Sürecin VMS (sanal bellek / mmap dahil) kullanımını MB cinsinden döndürür."""
    try:
        import psutil
        return psutil.Process().memory_info().vms / (1024 * 1024)
    except Exception:
        return None


def _disk_gb(model_path):
    """Modelin diskteki toplam boyutu (shard'lıysa tüm shard'ları toplar)."""
    p = Path(model_path)
    if "-00001-of-" in p.name:
        prefix = p.name.split("-00001-of-")[0]
        total = sum(f.stat().st_size for f in p.parent.glob(prefix + "*"))
    else:
        total = p.stat().st_size
    return total / (1024 ** 3)


def _model_yukle(model_path):
    print(f"\n>>> Model yükleniyor: {Path(model_path).name}", flush=True)
    t0 = time.time()
    llm = Llama(model_path=str(model_path), n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)
    print(f"    Yüklendi ({time.time() - t0:.1f} sn)", flush=True)
    return llm


def _model_serbest(llm):
    try:
        llm.close()
    except Exception:
        pass
    del llm
    gc.collect()


def _soru_calistir(llm, etiket):
    sonuc = []
    for i, soru in enumerate(SORULAR):
        t0 = time.time()
        cevap, kaynak, dogrulandi = generate_answer(llm, soru)
        sure = time.time() - t0
        sonuc.append({"soru": soru, "cevap": cevap, "sure": sure,
                      "kaynak": kaynak, "dogrulandi": dogrulandi})
        print(f"    [{etiket}] ({sure:.1f} sn) {soru[:44]}...", flush=True)
    return sonuc


def _anahtar_durum(i, cevap):
    """Bir sorunun cevabında kontrol anahtarlarının hangilerinin geçtiğini döndürür.

    (bulunan_listesi, tamam_mi) döndürür. tamam_mi: tüm anahtarlar geçiyorsa True.
    KONTROL'de yoksa (None, None) döner.
    """
    anahtarlar = KONTROL.get(i)
    if not anahtarlar:
        return None, None
    bulunan = [a for a in anahtarlar if a in cevap]
    return bulunan, len(bulunan) == len(anahtarlar)


def _anahtar_kisa(i, cevap):
    """Kontrol anahtarlarını kısa/kompakt biçimde yazar: "0.85✓ 1.0✓" gibi."""
    bulunan, _ = _anahtar_durum(i, cevap)
    if bulunan is None:
        return "—"
    anahtarlar = KONTROL[i]
    parcalar = []
    for a in anahtarlar:
        parcalar.append(f"{a}{'✓' if a in cevap else '✗'}")
    return " ".join(parcalar)


def main():
    print("Embedding modeli önceden yükleniyor (süre ölçümünü kirletmesin)...", flush=True)
    load_embedding_model()
    print("Embedding hazır.\n", flush=True)

    rss_bos = _rss_mb()
    vms_bos = _vms_mb()

    # Her modelin sonuçları: {etiket: {"sonuclar": [...], "rss": ..., "vms": ..., "disk": ..., "hata": ...}}
    tum = {}

    for idx, (etiket, yol) in enumerate(MODELLER, start=1):
        print("=" * 70, flush=True)
        print(f"AŞAMA {idx}/{len(MODELLER)}: {etiket} model", flush=True)
        print("=" * 70, flush=True)
        try:
            llm = _model_yukle(yol)
        except Exception as e:
            print(f"\n[{etiket} YÜKLENEMEDİ] {type(e).__name__}: {e}", flush=True)
            tum[etiket] = {"hata": str(e), "sonuclar": [], "rss": None, "vms": None,
                           "disk": _disk_gb(yol) if Path(yol).exists() else None}
            continue

        rss = _rss_mb()
        vms = _vms_mb()
        s = _soru_calistir(llm, etiket)
        _model_serbest(llm)
        print(f"{etiket} serbest bırakıldı.", flush=True)

        tum[etiket] = {"sonuclar": s, "rss": rss, "vms": vms,
                       "disk": _disk_gb(yol), "hata": None}

    # --- Rapor ---
    print("\n" + "#" * 78, flush=True)
    print("ÜÇLÜ KARŞILAŞTIRMA RAPORU (1.5B / 3B / 7B)", flush=True)
    print("#" * 78, flush=True)

    # Bellek özeti
    print("\n--- Bellek ---", flush=True)
    if rss_bos and vms_bos:
        print(f"  Boş süreç RSS        : {rss_bos:.0f} MB", flush=True)
        for etiket, _ in MODELLER:
            r = tum[etiket]
            if r["rss"] is not None:
                print(f"  {etiket:>4} yüklenince RSS : {r['rss']:.0f} MB  (+{r['rss'] - rss_bos:.0f} MB)", flush=True)
                print(f"  {etiket:>4} yüklenince VMS : {r['vms']:.0f} MB  (+{r['vms'] - vms_bos:.0f} MB)", flush=True)
            else:
                print(f"  {etiket:>4} yüklenince RSS : — (yüklenemedi)", flush=True)
    else:
        print("  (psutil kurulu değil; bellek ölçülemedi)", flush=True)
    for etiket, _ in MODELLER:
        d = tum[etiket]["disk"]
        if d is not None:
            print(f"  {etiket:>4} disk boyutu     : {d:.2f} GB", flush=True)

    # Soru bazında özet tablo
    print("\n--- Soru bazında özet ---", flush=True)
    baslik = f"{'Soru':<38} | {'1.5B sn':>7} | {'3B sn':>7} | {'7B sn':>7}"
    print(baslik, flush=True)
    print("-" * len(baslik), flush=True)
    for i, soru in enumerate(SORULAR):
        sureler = []
        for etiket, _ in MODELLER:
            r = tum[etiket]
            if r["sonuclar"]:
                sureler.append(f"{r['sonuclar'][i]['sure']:>7.1f}")
            else:
                sureler.append(f"{'—':>7}")
        print(f"{soru[:37]:<38} | {sureler[0]} | {sureler[1]} | {sureler[2]}", flush=True)

    print("\n--- Doğruluk anahtarları (bulunan anahtar) ---", flush=True)
    print(f"{'Soru':<38} | {'1.5B':<22} | {'3B':<22} | {'7B':<22}", flush=True)
    print("-" * 108, flush=True)
    for i, soru in enumerate(SORULAR):
        if i not in KONTROL:
            continue
        anahtarlar = KONTROL[i]
        hucreler = []
        for etiket, _ in MODELLER:
            r = tum[etiket]
            if not r["sonuclar"]:
                hucreler.append("—")
                continue
            cevap = r["sonuclar"][i]["cevap"]
            bulunan = [a for a in anahtarlar if a in cevap]
            hucreler.append(" ".join(f"{a}{'✓' if a in bulunan else '✗'}" for a in anahtarlar))
        print(f"{soru[:37]:<38} | {hucreler[0]:<22} | {hucreler[1]:<22} | {hucreler[2]:<22}", flush=True)

    # Cevap detayları
    for i, soru in enumerate(SORULAR):
        print(f"\n=== S{i+1}: {soru}", flush=True)
        for etiket, _ in MODELLER:
            r = tum[etiket]
            if not r["sonuclar"]:
                print(f"  [{etiket}] (yüklenemedi)", flush=True)
                continue
            o = r["sonuclar"][i]
            print(f"  [{etiket}] ({o['sure']:.1f} sn, {o['kaynak']}, doğrulandı={o['dogrulandi']}):", flush=True)
            print("    " + o["cevap"].replace("\n", "\n    "), flush=True)

    # JSON çıktısı (rapor için; süreler sn, bellek MB)
    json_verisi = {}
    for etiket, _ in MODELLER:
        r = tum[etiket]
        json_verisi[etiket] = {
            "hata": r["hata"],
            "disk_gb": round(r["disk"], 3) if r["disk"] is not None else None,
            "rss_mb": round(r["rss"], 1) if r["rss"] is not None else None,
            "vms_mb": round(r["vms"], 1) if r["vms"] is not None else None,
            "sonuclar": [
                {
                    "soru": o["soru"],
                    "cevap": o["cevap"],
                    "sure_sn": round(o["sure"], 2),
                    "kaynak": o["kaynak"],
                    "dogrulandi": o["dogrulandi"],
                    "anahtar": _anahtar_kisa(i, o["cevap"]) if (i in KONTROL) else "—",
                }
                for i, o in enumerate(r["sonuclar"])
            ],
        }
    JSON_CIKTI.write_text(json.dumps(json_verisi, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSonuçlar JSON olarak yazıldı: {JSON_CIKTI}", flush=True)


if __name__ == "__main__":
    main()
