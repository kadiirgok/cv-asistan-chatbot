# -*- coding: utf-8 -*-
"""
1.5B ve 7B modellerini aynı 4 test sorusu üzerinde karşılaştırır.

Aynı prompt/parametreler (max_tokens=150, temperature=0.1, repeat_penalty=1.3,
n_ctx=3072, n_threads=4) ile iki modeli sırayla çalıştırır, cevapları ve
süreleri kaydeder, yan yana karşılaştırır. Bellek kullanımını da mümkünse
(psutil ile) ölçer. Production'ı (api.py) DEĞİŞTİRMEZ; sadece ölçüm içindir.
"""

import gc
import time
from pathlib import Path

from llama_cpp import Llama

from rag import (
    MODEL_PATH_15B,
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

# Q1 (indeks 0) ve Q4 (indeks 3) için "doğru olguyu içeriyor mu" kontrol anahtarları
KONTROL = {
    0: ["0.85", "1.0"],            # Q1: hit-rate değerleri
    3: ["IQRClipper", "sızıntı"],  # Q4: veri sızıntısı düzeltmesi / IQRClipper
}

N_CTX = 3072
N_THREADS = 4


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


def main():
    print("Embedding modeli önceden yükleniyor (süre ölçümünü kirletmesin)...", flush=True)
    load_embedding_model()
    print("Embedding hazır.\n", flush=True)

    rss_bos = _rss_mb()
    vms_bos = _vms_mb()

    # 1) 1.5B
    print("=" * 70, flush=True)
    print("AŞAMA 1/2: 1.5B model", flush=True)
    print("=" * 70, flush=True)
    llm15 = _model_yukle(MODEL_PATH_15B)
    rss15 = _rss_mb()
    vms15 = _vms_mb()
    s15 = _soru_calistir(llm15, "1.5B")
    _model_serbest(llm15)
    print("1.5B serbest bırakıldı.", flush=True)

    # 2) 7B
    print("\n" + "=" * 70, flush=True)
    print("AŞAMA 2/2: 7B model", flush=True)
    print("=" * 70, flush=True)
    try:
        llm7 = _model_yukle(MODEL_PATH_7B)
    except Exception as e:
        print(f"\n[7B YÜKLENEMEDİ] {type(e).__name__}: {e}", flush=True)
        print("Büyük olasılıkla bellek yetmedi (RAM ~7.7 GB; 7B q4 ~4.4 GB + ek yük).", flush=True)
        _rapor_tek_model(s15, "1.5B")
        return
    rss7 = _rss_mb()
    vms7 = _vms_mb()
    s7 = _soru_calistir(llm7, "7B")
    _model_serbest(llm7)

    # --- Rapor ---
    print("\n" + "#" * 70, flush=True)
    print("KARŞILAŞTIRMA RAPORU", flush=True)
    print("#" * 70, flush=True)

    print("\n--- Bellek ---", flush=True)
    if rss_bos and vms_bos:
        print(f"  Boş süreç RSS              : {rss_bos:.0f} MB", flush=True)
        print(f"  1.5B yüklenince RSS        : {rss15:.0f} MB  (+{rss15 - rss_bos:.0f} MB)", flush=True)
        print(f"  7B yüklenince RSS          : {rss7:.0f} MB  (+{rss7 - rss_bos:.0f} MB)", flush=True)
        print(f"  1.5B yüklenince VMS        : {vms15:.0f} MB  (+{vms15 - vms_bos:.0f} MB)", flush=True)
        print(f"  7B yüklenince VMS          : {vms7:.0f} MB  (+{vms7 - vms_bos:.0f} MB)", flush=True)
    else:
        print("  (psutil kurulu değil; bellek ölçülemedi)", flush=True)
    print(f"  1.5B disk boyutu          : {_disk_gb(MODEL_PATH_15B):.2f} GB", flush=True)
    print(f"  7B disk boyutu (tüm shard): {_disk_gb(MODEL_PATH_7B):.2f} GB", flush=True)

    print("\n--- Soru bazında özet ---", flush=True)
    print(f"{'Soru':<42} | {'1.5B sn':>7} | {'7B sn':>7} | {'1.5B anahtar':>13} | {'7B anahtar':>13}", flush=True)
    print("-" * 96, flush=True)
    for i, soru in enumerate(SORULAR):
        anahtarlar = KONTROL.get(i, [])
        if anahtarlar:
            k15 = "evet" if any(a in s15[i]["cevap"] for a in anahtarlar) else "hayır"
            k7 = "evet" if any(a in s7[i]["cevap"] for a in anahtarlar) else "hayır"
        else:
            k15 = k7 = "—"
        print(f"{soru[:41]:<42} | {s15[i]['sure']:>7.1f} | {s7[i]['sure']:>7.1f} | {k15:>13} | {k7:>13}", flush=True)

    for i, soru in enumerate(SORULAR):
        print(f"\n=== S{i+1}: {soru}", flush=True)
        print(f"  [1.5B] ({s15[i]['sure']:.1f} sn, {s15[i]['kaynak']}, doğrulandı={s15[i]['dogrulandi']}):", flush=True)
        print("    " + s15[i]["cevap"].replace("\n", "\n    "), flush=True)
        print(f"  [7B] ({s7[i]['sure']:.1f} sn, {s7[i]['kaynak']}, doğrulandı={s7[i]['dogrulandi']}):", flush=True)
        print("    " + s7[i]["cevap"].replace("\n", "\n    "), flush=True)


def _rapor_tek_model(sonuc, etiket):
    """7B yüklenemediğinde tek modelin sonuçlarını basar."""
    print(f"\n--- {etiket} sonuçları (7B yüklenemedi) ---", flush=True)
    for i, soru in enumerate(SORULAR):
        print(f"\n=== S{i+1}: {soru}", flush=True)
        print(f"  [{sonuc[i]['sure']:.1f} sn] {sonuc[i]['cevap']}", flush=True)


if __name__ == "__main__":
    main()
