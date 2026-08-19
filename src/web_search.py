# -*- coding: utf-8 -*-
"""
Web arama fallback modülü.

Local RAG DB'de cevap bulunamadığında devreye girer: ddgs (DuckDuckGo) ile
soruyu internette arar ve en alakalı sonuçların başlık + özet kısmını tek bir
string olarak döndürür. API key gerektirmez; internet yoksa, timeout olursa ya
da arama başka bir sebeple patlarsa boş string döner ve hatayı loglar — programı
çökertmez.
"""

import logging

from ddgs import DDGS

logger = logging.getLogger(__name__)


def web_search_araci(soru: str, max_sonuc: int = 2) -> str:
    """Soruyu DuckDuckGo'da arar, en alakalı `max_sonuc` sonucu birleştirip döndürür.

    Her sonuç "Kaynak N: <title>\\nÖzet: <body>" biçiminde numaralandırılarak
    eklenir; sonuçlar aralarında boş satır olacak şekilde birleştirilir. Arama
    başarısız olursa (internet yok, timeout, istisna) boş string döner ve hatayı loglar.

    Not: embedding modelimiz Türkçe odaklı olduğu için region="tr-tr" ile Türkçe
    sonuçlar istenir; bu, sonucun ileride local'de tekrar bulunmasını kolaylaştırır.
    """
    try:
        with DDGS() as ddgs:
            sonuclar = ddgs.text(soru, max_results=max_sonuc, region="tr-tr")
    except Exception as exc:  # noqa: BLE001 — ağ/bağlantı hatalarını yutup logla
        logger.warning("Web araması başarısız (soru: %r): %s", soru, exc)
        return ""

    if not sonuclar:
        return ""

    parcalar = []
    kaynak_no = 0
    for r in sonuclar:
        baslik = (r.get("title") or "").strip()
        ozet = (r.get("body") or "").strip()
        if not baslik and not ozet:  # tamamen boş sonucu atla
            continue
        kaynak_no += 1
        parcalar.append(f"Kaynak {kaynak_no}: {baslik}\nÖzet: {ozet}")

    return "\n\n".join(parcalar)
