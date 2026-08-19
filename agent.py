# -*- coding: utf-8 -*-
"""
Agent katmanı (function-calling benzeri araç yönlendirme).

Qwen2.5-1.5B-Instruct GGUF, llama-cpp-python üzerinden OpenAI-uyumlu
yapılandırılmış `tool_calls` alanı ÜRETMEZ (bkz. test_function_calling.py):
aracı düz metin olarak, Qwen'in kendi `<tool_call>...</tool_call>` formatında
döndürür. Bu modül o çıktıyı parse eder, ilgili Python fonksiyonunu çalıştırır
ve sonucu modele geri vererek nihai cevabı üretir.

Akış (agent_yanit):
  1. Model `tools=TOOLS` ile çağrılır; ya doğrudan cevap ya da <tool_call> döner.
  2. <tool_call> yoksa -> cevap olduğu gibi döndürülür (basit sorular).
  3. <tool_call> varsa -> araç adı eşleştirilir, ilgili fonksiyon çalıştırılır,
     sonucu geçmişe eklenir ve model bu kez tools VERİLMEDEN yeniden çağrılır
     (nihai cevap istenir; tekrar araç çağrısı istenmez).
  4. Sonsuz döngüye karşı en fazla MAX_TUR araç turu denenir.

Not: api.py henüz bu modüle bağlanmadı; önce kendi başına doğru çalıştığı
test_agent.py ile doğrulanıyor.
"""

import json
import logging
import re
import time
from pathlib import Path

from llama_cpp import Llama

from rag import retrieval_araci
from web_search import web_search_araci

logger = logging.getLogger(__name__)

# Model dosyasının tam yolu (src/ klasörünün bir üstündeki models/)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Sonsuz döngü koruması: en fazla bu kadar araç turu denenir.
MAX_TUR = 3

# --- Araç şemaları (OpenAI function-calling formatında) ---
# Açıklamalar yönlendirmeyi belirler: retrieval önce denenmeli, web yalnızca
# local'de bulunamayan güncel/genel bilgi için kullanılmalı.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieval_araci",
            "description": (
                "Yerel veritabanındaki kendi dökümanlarımda arama yapar. "
                "Kendi dökümanlarımla ilgili bir soruda ÖNCE bunu dene."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "soru": {
                        "type": "string",
                        "description": "Yerel veritabanında aranacak soru/sorgu metni.",
                    }
                },
                "required": ["soru"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_araci",
            "description": (
                "Yerel veritabanında bulunamayan güncel veya genel bilgi için "
                "internette arama yapar. Yerel dökümanlarda cevap yoksa bunu kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "soru": {
                        "type": "string",
                        "description": "İnternette aranacak soru/sorgu metni.",
                    }
                },
                "required": ["soru"],
            },
        },
    },
]

# Araç adı -> Python fonksiyonu eşlemesi. İki fonksiyon da `soru` parametresi alır
# ve sonucu string olarak döndürür (retrieval_araci bağlam, web_search_araci özet).
# retrieval_araci artık (context, source) çifti döndürdüğü için ayrı bir sarmalayıcı
# kullanılır (araç arayüzü string bekler; source yalnızca debug içindir).
def _retrieval_araci_arac(soru: str) -> str:
    """retrieval_araci'yi araç arayüzüne uyarlar: yalnızca bağlam string'ini döndürür."""
    context, _source = retrieval_araci(soru)
    return context


ARAC_FONKSIYONLARI = {
    "retrieval_araci": _retrieval_araci_arac,
    "web_search_araci": web_search_araci,
}

# Model, aracı şu formatta döndürür:
#   <tool_call>
#   {{"name": "...", "arguments": {"soru": "..."}}
#   </tool_call>
# (Qwen bazen JSON'u çift süslü parantezle {{...}} sarar; parse sırasında indirgenir.)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# Model global değişkende tutulur; ilk kullanımda bir kez yüklenir (her çağrıda
# 1.1 GB modeli yeniden yüklemek çok yavaş olur). rag.py'deki embedding modeline
# benzer bir önbellekleme.
_llm = None


def _get_llm() -> Llama:
    """Modeli ilk çağrıda yükler, sonraki çağrılarda aynı nesneyi döndürür."""
    global _llm
    if _llm is None:
        print("Model yükleniyor (agent)...")
        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_threads=4,
            verbose=False,
        )
        print("Model hazır.")
    return _llm


def _json_coz(metin: str):
    """Metni JSON olarak çözer; başarısızsa None döner (çökmez, loglamaz).

    Loglama çağıran tarafta yapılır; böylece esnek parantez kırpma sırasında
    her aday için ayrı ayrı uyarı basılmaz.
    """
    try:
        return json.loads(metin)
    except (json.JSONDecodeError, ValueError):
        return None


def _json_coz_esnek(metin: str):
    """Fazladan süslü parantezleri tolere ederek JSON çözer.

    Qwen, tool_call JSON'unu bazen fazladan süslü parantezle sarar. Gözlenen
    örnekler: `{{"name":...}` (başta bir fazla `{`, sonda normal) ya da
    `{{"name":...}}` (iki yanda da fazla). Bu yardımcı baştan ve sondan 0..3
    arası `{`/`}` kırpıp ilk başarılı çözümü döndürür; hiçbiri olmazsa None.
    """
    metin = metin.strip()
    for on_kirp in range(4):       # baştan kaç `{` kırpılacağı (0..3)
        for arka_kirp in range(4):  # sondan kaç `}` kırpılacağı (0..3)
            aday = metin[on_kirp:]
            if arka_kirp:
                aday = aday[:-arka_kirp]
            veri = _json_coz(aday)
            if veri is not None:
                return veri
    return None


def parse_tool_call(model_ciktisi: str) -> dict | None:
    """Model çıktısından <tool_call>...</tool_call> bloğunu ayıklayıp parse eder.

    Başarılıysa {"name": ..., "arguments": {...}} döner; <tool_call> bloğu yoksa
    ya da JSON bozuk/eksikse None döner (çökmez, yalnızca loglar). arguments alanı
    model tarafından string olarak da verilebilir; o durumda yeniden çözülür.
    """
    if not model_ciktisi:
        return None

    eslesme = _TOOL_CALL_RE.search(model_ciktisi)
    if not eslesme:
        return None

    ham = eslesme.group(1).strip()
    veri = _json_coz_esnek(ham)

    if not isinstance(veri, dict):
        logger.warning("Tool_call içeriği JSON olarak çözülemedi: %.200r", ham)
        return None

    name = veri.get("name")
    if not name:
        logger.warning("Tool_call JSON'unda 'name' alanı yok: %r", veri)
        return None

    arguments = veri.get("arguments") or {}
    if isinstance(arguments, str):  # model arguments'i string verdi -> yeniden çöz
        arguments = _json_coz_esnek(arguments) or {}
    if not isinstance(arguments, dict):
        arguments = {}

    return {"name": name, "arguments": arguments}


def _calistir_arac(name: str, arguments: dict) -> str:
    """Araç adını ilgili Python fonksiyonuna eşler ve çalıştırır.

    Bilinmeyen araç adı, eksik parametre veya araç içindeki bir hata programı
    çökertmez; açıklayıcı bir string döndürülür ve durum loglanır.
    """
    fn = ARAC_FONKSIYONLARI.get(name)
    if fn is None:
        logger.warning("Bilinmeyen araç isteği: %r", name)
        return f"(Bilinmeyen araç: {name})"

    soru = str(arguments.get("soru", "")).strip()
    if not soru:
        logger.warning("Araç %r 'soru' parametresi olmadan çağrıldı.", name)
        return "(Araç çağrısında 'soru' parametresi eksik veya boş.)"

    try:
        sonuc = fn(soru)
    except Exception as exc:  # noqa: BLE001 — araç hatası ajanı çökertmesin
        logger.warning("Araç %r çalışırken hata: %s", name, exc)
        return f"(Araç {name} çalıştırılırken hata oluştu.)"

    sonuc = (sonuc or "").strip()
    return sonuc if sonuc else "(Araç sonuç döndürmedi.)"


def agent_yanit(soru: str) -> dict:
    """Soruya, araç yönlendirmesiyle cevap verir ve özet bir sözlük döndürür.

    Dönüş:
      {"cevap": str, "kullanilan_arac": str | None, "sure_saniye": float}

    `kullanilan_arac` modelin çağırdığı araç adıdır (hiç çağırmadıysa None);
    şeffaflık ve debug içindir. Birden fazla araç çağrılırsa "+" ile birleştirilir.
    """
    # Model yükü (ilk çağrıda tek seferlik, ~1.1 GB) cevap süresine dahil edilmez;
    # böylece tüm çağrılarda "sure_saniye" yalnızca gerçek çıkarım süresini ölçer.
    llm = _get_llm()
    t0 = time.time()
    kullanilan_araclar: list[str] = []
    cevap = ""

    # Sohbet geçmişi; ilk mesaj kullanıcının sorusudur. Araç sonucu buraya eklenir.
    gecmis = [{"role": "user", "content": soru}]

    for tur in range(1, MAX_TUR + 1):
        # Yalnızca İLK turda tools verilir; araç çağrısı yapıldıktan sonra tools
        # verilmez ki model nihai cevabı üretsin, tekrar araç çağırmasın.
        kwargs = {
            "messages": gecmis,
            "max_tokens": 512,
            "temperature": 0.0,  # deterministik yönlendirme + tekrarlanabilir cevap
        }
        if tur == 1:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"

        response = llm.create_chat_completion(**kwargs)
        content = (response["choices"][0]["message"].get("content") or "").strip()

        parsed = parse_tool_call(content)
        if parsed is None:
            # Araç çağrısı yok -> model doğrudan cevap verdi, olduğu gibi al.
            cevap = content
            break

        # Araç çağrısı var -> eşleştir, çalıştır, sonucu geçmişe ekle.
        name = parsed["name"]
        kullanilan_araclar.append(name)
        sonuc = _calistir_arac(name, parsed["arguments"])

        gecmis.append({"role": "assistant", "content": content})
        gecmis.append({
            "role": "user",
            "content": (
                f"Araç ({name}) şu sonucu döndürdü:\n{sonuc}\n\n"
                "Bu sonuca dayanarak kullanıcının sorusunu yanıtla."
            ),
        })

    if not cevap:
        cevap = "(Nihai cevap üretilemedi; maksimum araç çağrısı turuna ulaşıldı.)"

    kullanilan_arac = "+".join(kullanilan_araclar) if kullanilan_araclar else None
    return {
        "cevap": cevap,
        "kullanilan_arac": kullanilan_arac,
        "sure_saniye": round(time.time() - t0, 3),
    }
