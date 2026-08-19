# -*- coding: utf-8 -*-
"""
RAG (Retrieval-Augmented Generation) iskeleti.

Bu modül şu işleri yapar:
1. Uzun metni örtüşmeli (overlap) parçalara böler           -> chunk_text
2. Klasördeki .txt dosyalarını okur                          -> load_documents_from_folder
3. Verilen metinleri vektöre çevirip ChromaDB'ye kalıcı yazar -> build_index
4. Bir soruya en yakın chunk'ları bulur                       -> retrieval_araci
5. Bağlam + soruyu modele sunup cevap üretir                  -> generate_answer

Embedding için Türkçe odaklı "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
modeli kullanılır (kalıcı seçim).
"""

import hashlib
import re
from collections import Counter
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from web_search import web_search_araci

# --- Embedding modeli (kalıcı seçim, sabit) ---
# Türkçe odaklı, NLI + STS (anlam benzerliği) verisiyle eğitilmiş model.
# MiniLM'e göre Türkçe'de daha iyi benzerlik sıralaması üretir; artık deneme
# değil, bu projenin kalıcı embedding modeli olarak sabitlenmiştir.
EMBEDDING_MODEL_NAME = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"

# --- ChromaDB persist klasörü ---
# Bu model için ayrılmış kalıcı klasör. Diğer chroma_db_* klasörleri referans
# için yerinde durur; silinmez.
PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db_emrecan")

# Vektörlerin tutulacağı koleksiyon adı
COLLECTION_NAME = "turkce_belgeler"

# --- Model dosyası yolları (1.5B / 3B / 7B geçişini tek noktadan yönetmek için) ---
# GGUF dosyaları models/ klasöründe tutulur. 7B q4_k_m resmi Qwen deposunda iki
# shard halindedir; llama.cpp ilk shard verildiğinde diğerini otomatik yükler,
# bu yüzden MODEL_PATH_7B ilk shard'ı gösterir. 1.5B ve 3B tek dosyadır. Üretim/
# sohbet hangi modeli kullanacağını MODEL_PATH sabitiyle seçer (tek satır değişir).
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH_15B = str(MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf")
MODEL_PATH_3B = str(MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf")
MODEL_PATH_7B = str(MODELS_DIR / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")

# Varsayılan model (production/sohbet için). 7B'ye geçmek için MODEL_PATH_7B yapın.
# 3B: 1.5B'ye göre belirgin şekilde daha iyi doğruluk; n_ctx=3072 ile test edildi,
# kalite kaybı olmadan çalışıyor. Production artık 3B kullanır.
MODEL_PATH = MODEL_PATH_3B

# Bağlam + soruyu modele sunacağımız prompt şablonu (local RAG için)
PROMPT_TEMPLATE = (
    "Aşağıdaki bağlamı kullanarak soruyu yanıtla. "
    "Bağlamda cevap yoksa bilmediğini söyle.\n\n"
    "Bağlam:\n{context}\n\n"
    "Soru: {soru}\n\n"
    "Kurallar:\n"
    "- Bağlamdaki sayıları, listeleri ve teknik terimleri AYNEN aktar, özetleyip kısaltma.\n"
    "- Cevabını kısa ve net tut, aynı cümleyi veya kelimeyi tekrar etme.\n"
    "- Bağlamda birden fazla ilgili değer/madde varsa hepsini sırayla listele, sadece birini seçme.\n"
    "- Cevabını tek kelime veya yarım cümle olarak değil, tam ve akıcı bir Türkçe cümleyle ver. "
    "Kısa ama doğal bir açıklama cümlesi kur — örneğin sadece 'Hit-rate@1=0.85' yazmak yerine, "
    "'BilgiTR projesinde Hit-rate@1 değeri 0.85, Hit-rate@3 ve Hit-rate@5 değerleri ise 1.0 "
    "olarak ölçüldü.' gibi bağlam veren bir cümle kur.\n"
    "- Telgraf gibi, sadece anahtar kelimelerden oluşan bir cevap verme; normal konuşma diliyle, "
    "tam cümlelerle yaz.\n"
    "Yanıt:"
)

# Web arama sonuçları için daha sıkı prompt şablonu. Web sonuçları gürültülü
# olduğu için local RAG'den ayrı tutulur: modele yalnızca kaynaklarda açıkça
# yazan bilgiyi kullanması, kendi önbilgisiyle karıştırmaması ve çelişkide
# uydurmaması söylenir. {context} zaten "Kaynak 1: ... Kaynak 2: ..." biçiminde
# numaralı gelir (bkz. web_search_araci).
WEB_PROMPT_TEMPLATE = (
    "Aşağıda internetten bulunan arama sonuçları var. SADECE bu sonuçlarda "
    "açıkça yazan bilgiyi kullan. Kendi önceden bildiğin bir bilgiyle karıştırma, "
    "tahmin yürütme.\n\n"
    "{context}\n\n"
    "Soru: {soru}\n\n"
    "KURAL: Cevabında geçen her yer, kişi ve tarih adı, yukarıdaki kaynaklarda "
    "aynen geçmiş olmalıdır. Kaynaklarda yazmayan hiçbir adı kullanma. Kaynaklar "
    "arasında net bir cevap yoksa veya çelişki varsa bunu açıkça belirt, uydurma.\n"
    "Yanıt:"
)

# --- Mesafe eşiği (distance threshold) ---
# ChromaDB varsayılan L2 (squared) mesafesini kullanır. emrecan modeli normalize
# edilmemiş (mean-pooled) vektörler ürettiği için mesafeler küçük değil, ~200-720
# aralığında çıkar. probe_distances.py'deki gerçek ölçümler (top-1):
#   alakalı chunk  : 201.74 (mercimek), 284.81 (Bursa), 341.77 (İzmir),
#                    375.15 (Richter), 461.36 (Bursa/d2)  -> en yükseği 461.36
#   alakasız chunk : 523.37 (ilk alakasız), 551.21 (d1'de Bursa), 648.89 (c sorusu)
# İki grubun arasındaki boşluk 461.36 ile 523.37 arasındadır; ortası ~492'dir.
# 490 hem en yüksek alakalıyı (461) içerir hem de tüm alakasızları (>=523) eler.
DEFAULT_MESAFE_ESIGI = 490.0

# Embedding modeli bir kez yüklenip önbelleğe alınır (her çağrıda yeniden yüklemek yavaş olur)
_model = None


def _get_embedding_model():
    """Embedding modelini ilk çağrıda yükler ve sonraki çağrılar için önbelleğe alır."""
    global _model
    if _model is None:
        print("Embedding modeli yükleniyor (ilk sefer biraz sürebilir)...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def load_embedding_model() -> None:
    """Embedding modelini önceden yükleyip önbelleğe alır (ilk isteği hızlandırmak için)."""
    _get_embedding_model()


def _get_collection():
    """ChromaDB istemcisini açıp koleksiyonu hazırlar (yoksa oluşturur)."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Uzun bir metni kelime bazında, belirtilen örtüşmeyle parçalara böler.

    overlap sayesinde parça sınırındaki bir cümle ikiye bölünmez ve bilgi kaybı azalır.
    """
    words = text.split()
    if not words:  # boş metin -> boş liste
        return []

    chunks = []
    step = max(1, chunk_size - overlap)  # bir sonraki parçaya ne kadar kayılacağı
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        # Son parçaya ulaştıysak döngüyü bitir (küçük artıkları ayrıca ekleme)
        if i + chunk_size >= len(words):
            break
    return chunks


def load_documents_from_folder(folder_path: str = "./data") -> list[dict]:
    """Belirtilen klasördeki tüm .txt dosyalarını okur ve liste olarak döndürür.

    Her dosya tek bir doküman olarak {"text": içerik, "source": dosya_adı}
    sözlüğü biçiminde döndürülür. "source" alanı, ileride cevabın hangi
    dosyadan geldiğini takip edebilmek için ChromaDB'ye metadata olarak yazılır.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Uyarı: '{folder_path}' klasörü bulunamadı.")
        return []

    documents = []
    # .txt dosyalarını alfabetik sırayla oku (her çalıştırmada aynı sonucu almak için)
    for txt_file in sorted(folder.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8").strip()
        if content:  # boş dosyaları atla
            documents.append({"text": content, "source": txt_file.name})
    return documents


def _chunk_id(text: str) -> str:
    """Chunk içeriğinden kararlı (deterministik) bir kimlik üretir.

    İçeriğe bağlı (content-addressable) id sayesinde aynı chunk tekrar
    indekslendiğinde aynı id'yi alır (upsert ile güncellenir), farklı
    dokümanlar ise farklı id alarak mevcut verinin YANINA eklenir (üzerine yazmaz).
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_index(documents: list) -> None:
    """Verilen metinleri chunk'lar, embed eder ve ChromaDB'ye kalıcı yazar.

    `documents` ya `list[str]` ya da `list[dict]` olabilir. dict durumunda
    "text" (metin), "source" (kaynak) ve isteğe bağlı "tarih" anahtarları kullanılır.
    """
    model = _get_embedding_model()
    collection = _get_collection()

    # Her dokümanı chunk'lara böl; her chunk'a içerik-hash id ve kaynak metadata ver
    ids, texts, metadatas = [], [], []
    for doc in documents:
        # Hem düz metin hem de {"text", "source", "tarih"} sözlüğü desteklenir
        if isinstance(doc, dict):
            text = doc.get("text", "")
            source = doc.get("source", "")
            tarih = doc.get("tarih", "")
        else:
            text = doc
            source = ""
            tarih = ""

        for chunk in chunk_text(text):
            ids.append(_chunk_id(chunk))  # içerik-hash id: mevcut veriyle çakışmaz
            texts.append(chunk)
            metadatas.append({"source": source, "tarih": tarih})

    if not texts:
        print("Uyarı: İndekslenecek boş olmayan metin bulunamadı.")
        return

    print(f"{len(texts)} chunk embedding'e çevriliyor...")
    # Tüm chunk'ları tek seferde embed et (tek tek yapmaktan çok daha hızlı)
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    # upsert: aynı id varsa günceller, yoksa ekler -> script güvenle tekrar çalıştırılabilir
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"İndeks tamamlandı: {len(texts)} chunk '{COLLECTION_NAME}' koleksiyonuna yazıldı.")


# --- Leksikal tie-break için Türkçe sık kelimeler (soru / bağlaç / edat) ---
# retrieval_araci'deki çoğunluk oyu berabere kaldığında, sorudaki "anlamlı"
# kelimeler source'un dosya adıyla/chunk içeriğiyle eşleştirilir. Bu küme,
# eşleşme sinyali vermeyen yaygın soru ve fonksiyon kelimelerini eler.
_SORU_STOPWORDS = {
    # soru kelimeleri
    "nedir", "nasıl", "hangi", "hangisi", "ne", "neden", "niye", "niçin",
    "kaç", "kim", "kime", "neyi", "nerede", "nereden", "nereye", "neresi",
    # bağlaç / edat / sık fonksiyon kelimeleri
    "bir", "ve", "veya", "ya", "da", "de", "ile", "için", "daha", "çok", "en",
    "bu", "şu", "o", "gibi", "kadar", "göre", "sonra", "önce", "artık", "hâlâ",
    "mı", "mi", "mu", "mü", "ise", "çünkü", "ama", "fakat", "ancak", "olarak",
    "olan", "olduğu", "var", "yok", "her", "bazı", "tüm", "hiç",
    "verdi", "veriyor", "olur", "oldu", "eder", "etmek", "olmak",
}


def _soru_kelimeleri(soru: str) -> list[str]:
    """Sorudaki anlamlı kelimeleri çıkarır (basit sezgisel yöntem).

    - Boşluklara göre böler, kenarlardaki noktalamayı temizler.
    - 3+ karakterli kelimeler tutulur.
    - Yaygın soru / bağlaç / edat kelimeleri (_SORU_STOPWORDS) elenir.
    """
    kelimeler = []
    for ham in soru.lower().split():
        t = ham.strip(".,;:!?()[]{}\"'")
        if len(t) >= 3 and t not in _SORU_STOPWORDS:
            kelimeler.append(t)
    return kelimeler


def _kompakt(metin: str) -> str:
    """Metni küçük harfe çevirip harf/rakam dışındaki karakterleri kaldırır.

    Leksikal eşleşmede boşluk/tire toleransı sağlar: "bilgi tr" -> "bilgitr",
    "telco-churn" -> "telcochurn", "bilgitr-proje.txt" -> "bilgitrprojetxt".
    """
    return re.sub(r"[^a-z0-9çğıöşü]", "", metin.lower())


def _source_anahtari(source: str) -> str:
    """Source dosya adından kompakt proje anahtarını çıkarır.

    "bilgitr-proje.txt" -> "bilgitr", "telco-churn-proje.txt" -> "telcochurn",
    "yorumtr-proje.txt" -> "yorumtr".
    """
    ad = source
    if ad.endswith(".txt"):
        ad = ad[:-4]
    if ad.endswith("-proje"):
        ad = ad[:-6]
    return _kompakt(ad)


def _mesafe_tie_break(adaylar: list[str], secilenler: list[tuple[str, str]]) -> str:
    """Beraberlikte mesafeye göre en önce gelen (en alakalı) chunk'ın source'u."""
    return min(adaylar, key=lambda s: next(i for i, (_, src) in enumerate(secilenler) if src == s))


def _lexikal_tie_break(adaylar: list[str], secilenler: list[tuple[str, str]],
                       soru: str) -> str:
    """Beraberlikteki source'ları soru kelimeleriyle eşleştirir.

    Sorudaki anlamlı kelimelerin, aday source'un dosya adında (örn. "BilgiTR" ->
    "bilgitr" -> "bilgitr-proje.txt") veya o source'a ait chunk metninde geçip
    geçmediğine bakar; en çok kelime eşleşen source'u seçer. Hiçbiri eşleşmezse
    mevcut davranışa (en düşük mesafeli chunk'ın source'u) geri döner.
    """
    kelimeler = _soru_kelimeleri(soru)
    if not kelimeler:
        return _mesafe_tie_break(adaylar, secilenler)

    soru_kompakt = _kompakt(soru)

    skorlar = {}
    for source in adaylar:
        dosya = source.lower()
        metin = " ".join(doc for doc, s in secilenler if s == source).lower()
        skor = sum(1 for k in kelimeler if k in dosya or k in metin)
        # Boşluk/tire toleransı: sorunun kompakt hali source'un proje anahtarını
        # içeriyorsa (örn. "bilgi tr" -> "bilgitr") güçlü bir sinyaldir.
        anahtar = _source_anahtari(source)
        if anahtar and len(anahtar) >= 4 and anahtar in soru_kompakt:
            skor += 100
        skorlar[source] = skor

    en_iyi = max(adaylar, key=lambda s: skorlar[s])
    if skorlar[en_iyi] > 0:
        return en_iyi
    return _mesafe_tie_break(adaylar, secilenler)


def _leksikal_fallback(soru: str, collection) -> tuple[str, str]:
    """Embedding eşiği geçemediğinde, soruyu source dosya adlarıyla eşleştirir.

    Boşluk/tire toleransıyla çalışır: "bilgi tr" -> "bilgitr" -> "bilgitr-proje.txt",
    "telco churn" -> "telcochurn" -> "telco-churn-proje.txt". En uzun (en spesifik)
    eşleşen proje anahtarının source'una ait tüm chunk'ları döndürür; hiçbiri
    eşleşmezse ("", "") döner.
    """
    soru_kompakt = _kompakt(soru)
    if not soru_kompakt:
        return "", ""

    veri = collection.get(include=["documents", "metadatas"])
    docs = veri.get("documents", [])
    metas = veri.get("metadatas", [])

    # source -> o source'a ait chunk'lar
    by_source: dict[str, list[str]] = {}
    for doc, meta in zip(docs, metas):
        source = (meta or {}).get("source", "")
        if source == "web" or not source:
            continue
        by_source.setdefault(source, []).append(doc)

    en_iyi_source = ""
    en_iyi_uzunluk = 0
    for source in by_source:
        anahtar = _source_anahtari(source)
        if len(anahtar) >= 4 and anahtar in soru_kompakt and len(anahtar) > en_iyi_uzunluk:
            en_iyi_source = source
            en_iyi_uzunluk = len(anahtar)

    if not en_iyi_source:
        return "", ""
    return "\n\n".join(by_source[en_iyi_source]), en_iyi_source


def retrieval_araci(soru: str, top_k: int = 4, mesafe_esigi: float = DEFAULT_MESAFE_ESIGI) -> tuple[str, str]:
    """Soruyu embed eder, en yakın top_k chunk'ı bulur ve (context, source) döndürür.

    Her chunk'ın mesafesi `mesafe_esigi` ile karşılaştırılır; mesafesi eşiğin
    ÜSTÜNDE olan (yani yeterince alakalı olmayan) chunk'lar elenir. Tümü elenirse
    ("", "") döner (ilgili bağlam yok demektir). Düşük mesafe = daha alakalı.

    Farklı dökümanlardan gelen chunk'lar birbirine karışmasın diye, eşiği geçen
    chunk'lar source (hangi dosyadan geldiği) bazında gruplanır ve en çok chunk'a
    sahip source (çoğunluk oyu) seçilir. Yalnızca bu source'a ait chunk'lar
    bağlama konur; diğer source'lardan gelen "yabancı" chunk'lar atılır. Eşitlik
    durumunda önce sorudaki anlamlı kelimelerle leksikal eşleşme denenir (dosya
    adı/chunk içeriği); hiçbiri eşleşmezse en düşük mesafeli chunk'ın source'u
    kazanır.

    Dönüş: (context, source). context yalnızca çoğunluk source'unun chunk'larından
    oluşan string; source bu dökümanın dosya adı (debug ve fallback mesajı için).
    """
    model = _get_embedding_model()
    collection = _get_collection()

    # Soruyu vektöre çevirip veritabanında benzerlik araması yap.
    # "web" source'lu chunk'lar (eski web aramalarının önbelleği) sorguya hiç dahil
    # edilmez; böylece top_k slotlarını doldurup gerçek local chunk'ları dışarıda bırakmazlar.
    query_embedding = model.encode([soru]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"source": {"$ne": "web"}},
    )

    retrieved_docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # Her chunk'ı eşikle filtrele: (doc, source) çiftleri olarak topla.
    # "web" source'lu chunk'lar (eski web aramalarının önbelleği) local retrieval'ı
    # kirlettiği için dışarıda bırakılır; bunlar yalnızca web fallback yolunda işe yarar.
    secilenler = []
    for doc, dist, meta in zip(retrieved_docs, distances, metadatas):
        source = (meta or {}).get("source", "")
        if source == "web":
            continue
        if dist <= mesafe_esigi:
            secilenler.append((doc, source))

    if not secilenler:
        # Embedding eşiği geçemediyse leksikal fallback dene: sorudaki proje adını
        # (boşluk/tire toleransıyla) source dosya adıyla eşleştir.
        return _leksikal_fallback(soru, collection)

    # Çoğunluk oyu: en çok chunk'a sahip source'u bul. Beraberlikte (birden fazla
    # source aynı chunk sayısına sahipse) leksikal tie-break uygulanır: sorudaki
    # anlamlı kelimeler dosya adı/chunk içeriğiyle eşleştirilir; hiçbiri eşleşmezse
    # en düşük mesafeli chunk'ın source'u kazanır.
    kaynak_sayaci = Counter(source for _, source in secilenler)
    max_sayi = max(kaynak_sayaci.values())
    adaylar = [s for s, n in kaynak_sayaci.items() if n == max_sayi]

    if len(adaylar) == 1:
        cogunluk_source = adaylar[0]
    else:
        cogunluk_source = _lexikal_tie_break(adaylar, secilenler, soru)

    # Yalnızca çoğunluk source'una ait chunk'ları bağlama koy.
    filtrelenmis = [doc for doc, s in secilenler if s == cogunluk_source]
    return "\n\n".join(filtrelenmis), cogunluk_source


def _llm_cevap(llm, context: str, soru: str, max_tokens: int,
               prompt_template: str = PROMPT_TEMPLATE, temperature: float = 0.3,
               repeat_penalty: float = 1.3) -> str:
    """Bağlam + soruyu şablona koyup modelden cevap üretir (ortak yardımcı).

    Local RAG için PROMPT_TEMPLATE, web sonuçları için WEB_PROMPT_TEMPLATE kullanılır.
    Web cevabında sıcaklık 0'a çekilir (greedy): kaynağa sadakat için yaratıcılık kısılır.
    `repeat_penalty` (varsayılan 1.3) aynı token dizisinin tekrar tekrar üretilmesini
    caydırır (llama.cpp sampling parametresi; küçük modellerde tekrar döngüsünü azaltır).
    """
    prompt = prompt_template.format(context=context, soru=soru)
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 — model çağrısı hata verirse üst katmana bildir
        raise RuntimeError(f"Model cevap üretirken hata oluştu: {exc}") from exc


# --- Sayısal tutarlılık doğrulama (sayısal uydurma güvenlik ağı) ---
# Model cevabındaki sayıların kaynak metindeki sayılarla eşleşip eşleşmediğini
# kontrol eder. Ondalık (1.47 / 1,47), yüzde (%20 / 20%) ve para birimli
# (1.47 TL / $5) biçimlerdeki sayısal DEĞERİ yakalar; % / $ / TL gibi işaretler
# değere katılmaz (karşılaştırma yalnızca rakam üzerinden yapılır). %100 kusursuz
# değildir; amaç açık sayısal uydurmaları yakalamaktır.
_SAYI_RE = re.compile(r"(?<![\w])([+-]?\d+(?:[.,]\d+)?)(?![\w])")

# İki sayının "çok yakın" sayılabilmesi için göreli tolerans (yuvarlama farkı).
_SAYI_TOLERANS = 0.02


def _sayi_normalize(token: str) -> str:
    """Sayı token'ını karşılaştırma için tek biçime indirir (virgül->nokta, sonda 0 kırp).

    Ör. "35,20" -> "35.2", "1453" -> "1453". Normalize edilemeyen token olduğu gibi döner.
    """
    t = token.strip().replace(",", ".")
    try:
        return f"{float(t):g}"
    except ValueError:
        return t


def _sayi_degeri(token: str) -> float | None:
    """Sayı token'ını float'a çevirir; çevrilemezse None döner."""
    try:
        return float(token.strip().replace(",", "."))
    except ValueError:
        return None


def _sayilar_yakin(a: float, b: float) -> bool:
    """İki sayıyı göreli toleransla karşılaştırır (yuvarlama farkını tolere eder)."""
    return abs(a - b) <= _SAYI_TOLERANS * max(abs(a), abs(b), 1.0)


def dogrula_sayisal_tutarlilik(cevap: str, kaynak_metin: str) -> bool:
    """Cevaptaki her sayının kaynak metinde karşılığı olup olmadığını kontrol eder.

    - Cevapta sayı yoksa True (sorun yok).
    - Cevaptaki bir sayı, kaynak metindeki hiçbir sayıyla tam ya da çok yakın
      eşleşmiyorsa False döner (muhtemel sayısal uydurma).
    """
    cevap_sayilari = _SAYI_RE.findall(cevap)
    if not cevap_sayilari:
        return True  # cevapta sayı yok -> kontrol edilecek şey yok

    kaynak_sayilari = _SAYI_RE.findall(kaynak_metin)
    if not kaynak_sayilari:
        return False  # cevapta sayı var ama kaynakta hiç yok -> uydurma

    kaynak_norm = {_sayi_normalize(s) for s in kaynak_sayilari}
    kaynak_degerler = [d for d in (_sayi_degeri(s) for s in kaynak_sayilari) if d is not None]

    for s in cevap_sayilari:
        if _sayi_normalize(s) in kaynak_norm:
            continue  # tam eşleşme
        deger = _sayi_degeri(s)
        if deger is not None and any(_sayilar_yakin(deger, k) for k in kaynak_degerler):
            continue  # çok yakın eşleşme (yuvarlama toleransı)
        return False  # kaynakta karşılığı olmayan sayı bulundu

    return True


# Birbirine "liste" sayılacak kadar yakın iki ondalıklı sayı arasındaki azami
# karakter mesafesi. "Hit-rate@1=0.85, Hit-rate@3=1.0" gibi kalıplar ~13 karakterlik
# boşluk üretir; farklı satırlardaki (ablation) sayılar ise 30+ karakterle ayrılır.
_LISTE_BOSLUK = 25


def _liste_sayilari(kaynak_metin: str) -> list[str]:
    """Kaynaktaki ilk 'liste'nin sayılarını döndürür (yoksa boş liste).

    "Liste", aynı cümle/madde içinde birbirine yakın (boşluk <= _LISTE_BOSLUK) 3 veya
    daha fazla ondalıklı sayının oluşturduğu gruptur (örn. "0.85, 1.0, 1.0"). Yalnızca
    ondalıklı sayılara bakılır; böylece "@1=0.85" kalıbındaki indeks (1) ve "27 makale"
    gibi dağınık tamsayılar liste tespitini kirletmez.
    """
    eslesmeler = list(_SAYI_RE.finditer(kaynak_metin))
    ondalik = [m for m in eslesmeler if "." in m.group(0) or "," in m.group(0)]
    if len(ondalik) < 3:
        return []

    for i in range(len(ondalik) - 2):
        grup = [ondalik[i]]
        for j in range(i + 1, len(ondalik)):
            bosluk = kaynak_metin[grup[-1].end(): ondalik[j].start()]
            if len(bosluk) <= _LISTE_BOSLUK:
                grup.append(ondalik[j])
            else:
                break
        if len(grup) >= 3:
            return [m.group(0) for m in grup]
    return []


def dogrula_eksik_deger(cevap: str, kaynak_metin: str) -> bool:
    """Kaynaktaki bir 'liste'nin cevapta eksik kalıp kalmadığını kontrol eder.

    dogrula_sayisal_tutarlilik'in TERSİ yönde çalışır ama artık YALNIZCA kaynaktaki
    gerçek "listelere" bakar: aynı cümle/madde içinde birbirine yakın (virgülle / art
    arda) gelen 3+ sayılık gruplar "liste" sayılır. Kaynaktaki dağınık, birbiriyle
    ilişkisiz tekil sayılar ("27 makale", "400 kelime" gibi) görmezden gelinir.

    - Kaynakta liste yoksa True (kontrol edilecek şey yok).
    - Listenin TÜM sayıları cevapta geçiyorsa True.
    - Listeden bazı sayılar cevapta eksikse (örn. 3 sayıdan yalnız 1'i) False.

    Kesin bir kural değil, sezgisel bir kontrol; amaç "modelin bir listeyi tek değere
    indirgediği" bariz durumları yakalamaktır, mükemmel olması gerekmez.
    """
    liste_sayilari = _liste_sayilari(kaynak_metin)
    if not liste_sayilari:
        return True  # kaynakta liste yok -> kontrol uygulama

    cevap_norm = {_sayi_normalize(s) for s in _SAYI_RE.findall(cevap)}
    liste_norm = {_sayi_normalize(s) for s in liste_sayilari}
    # Listenin tüm sayıları cevapta geçiyorsa True; bazıları eksikse False.
    return liste_norm <= cevap_norm


# --- Konu uyumu doğrulaması ---
# "bilgi tr hakkında detaylı bilgi ver" gibi soruların, retrieval/web sonucunda
# alakasız bir konuya (örn. "Türkiye") kaymasını engeller. Bilinen proje adlarının
# kompakt anahtarlarını önbellekte tutar (her çağrıda koleksiyonu açmamak için).
_bilinen_anahtarlar_ondabellek: list[str] | None = None


def _bilinen_anahtarlar() -> list[str]:
    """Koleksiyondaki bilinen proje adlarının kompakt anahtarlarını döndürür (önbellekli)."""
    global _bilinen_anahtarlar_ondabellek
    if _bilinen_anahtarlar_ondabellek is None:
        try:
            veri = _get_collection().get(include=["metadatas"])
            sources = {(m or {}).get("source", "") for m in veri.get("metadatas", [])}
            _bilinen_anahtarlar_ondabellek = [
                _source_anahtari(s) for s in sources if s and s != "web"
            ]
        except Exception:
            _bilinen_anahtarlar_ondabellek = []
    return _bilinen_anahtarlar_ondabellek


def dogrula_konu_uyumu(soru: str, kullanilan_context: str) -> bool:
    """Sorunun konusu ile kullanılan context'in uyuşup uyuşmadığını kontrol eder.

    - Sorudaki anlamlı kelimelerin context'te geçip geçmediğine bakar.
    - Boşluk/tire toleransıyla, soruda bilinen bir proje adı geçiyorsa (örn.
      "bilgi tr" -> "bilgitr"), o adın context'te geçmesini bekler; geçmiyorsa
      konu uyuşmuyor demektir (False). Proje adı yoksa en az bir anlamlı kelimenin
      örtüşmesi yeterlidir.
    """
    kelimeler = _soru_kelimeleri(soru)
    if not kelimeler:
        return True  # kontrol edilecek kelime yok

    soru_kompakt = _kompakt(soru)
    ctx_kompakt = _kompakt(kullanilan_context)

    # Soruda bilinen bir proje adı geçiyorsa, o adın context'te geçmesi gerekir.
    for anahtar in _bilinen_anahtarlar():
        if len(anahtar) >= 4 and anahtar in soru_kompakt:
            return anahtar in ctx_kompakt

    # Proje adı yoksa: en az bir anlamlı kelime örtüşmeli.
    return any(_kompakt(k) in ctx_kompakt for k in kelimeler)


def _dogrulanmamis_fallback(kaynak_metin: str, source: str = "") -> str:
    """Doğrulanamayan model cevabı yerine kaynağı olduğu gibi (kısaltılmış) döndürür.

    `source` dökümanın dosya adıdır; fallback mesajına hangi dosyadan geldiği de
    eklenir (şeffaflık için). Web yolunda "web" etiketi kullanılır.
    """
    etiket = source or "bilinmiyor"
    return (f"Modelin ürettiği cevap doğrulanamadı, kaynaktaki ilgili bilgi "
            f"(Kaynak: {etiket}): {kaynak_metin[:300].strip()}")


def generate_answer(llm, soru: str, top_k: int = 4, max_tokens: int = 220,
                    mesafe_esigi: float = DEFAULT_MESAFE_ESIGI) -> tuple[str, str, bool]:
    """RAG akışını çalıştırır ve (cevap, kaynak, dogrulandi) üçlüsünü döndürür.

    Akış:
      1. Local DB'de retrieval_araci ile ara (mevcut mesafe eşiğiyle).
      2. Local'de alakalı bağlam VARSA -> o bağlamla cevap üret, kaynak="local"
         (web'e hiç gidilmez, hız için).
      3. Local'de bağlam YOKSA -> web_search_araci ile internette ara:
         - Web'de de sonuç yoksa -> "bilmiyorum" cevabı, kaynak="none".
         - Web'de sonuç varsa -> o bağlamla cevap üret, kaynak="web".

    Hem local hem web yolunda üretilen cevap ÜÇ doğrulamayla kontrol edilir:
    `dogrula_konu_uyumu` (sorunun konusu bağlamla uyuşuyor mu -> yoksa yanlış
    retrieval/alakasız web sonucu), `dogrula_sayisal_tutarlilik` (cevapta kaynakta
    karşılığı olmayan sayı var mı -> uydurma) ve `dogrula_eksik_deger` (kaynaktaki
    sayılardan bazıları cevapta eksik mi -> liste tek değere indirgenmiş). Konu
    uyuşmuyorsa "Bu konuda elimde güvenilir bilgi yok." döner. Sayısal kontrollerden
    biri False ise cevap KULLANILMAZ; çoğunluk source'una ait kaynak metnin ilk 300
    karakteri, dosya etiketiyle ("Kaynak: ...") döndürülür ve `dogrulandi=False` olur.

    `kaynak` "local", "web" veya "none"; `dogrulandi` sayısal doğrulamanın sonucu
    (debug ve şeffaflık içindir).

    `llm` parametresi `create_chat_completion` metodu olan herhangi bir nesne olabilir;
    böylece bu modül llama_cpp'e doğrudan bağımlı olmadan kalır.
    """
    # 1) Local DB'de bağlamı bul (mevcut mesafe eşiğiyle + çoğunluk-source filtresi)
    context, source = retrieval_araci(soru, top_k=top_k, mesafe_esigi=mesafe_esigi)

    if context.strip():
        # Konu uyuşmuyorsa (yanlış retrieval ya da alakasız bağlam) cevabı üretme.
        if not dogrula_konu_uyumu(soru, context):
            return "Bu konuda elimde güvenilir bilgi yok.", "none", False

        try:
            cevap = _llm_cevap(llm, context, soru, max_tokens, temperature=0.2)
        except Exception:
            return "Model şu anda yanıt üretemiyor (teknik bir sorun oluştu).", "local", False

        dogrulandi = (dogrula_sayisal_tutarlilik(cevap, context)
                      and dogrula_eksik_deger(cevap, context))
        if not dogrulandi:
            cevap = _dogrulanmamis_fallback(context, source)
        return cevap, "local", dogrulandi

    # 2) Local'de yoksa web'e git (fallback)
    web_context = web_search_araci(soru)
    if not web_context.strip():
        return "Bu konuda elimde bilgi yok.", "none", True

    if not dogrula_konu_uyumu(soru, web_context):
        return "Bu konuda elimde güvenilir bilgi yok.", "none", False

    try:
        cevap = _llm_cevap(llm, web_context, soru, max_tokens,
                           prompt_template=WEB_PROMPT_TEMPLATE, temperature=0.0)
    except Exception:
        return "Model şu anda yanıt üretemiyor (teknik bir sorun oluştu).", "web", False

    dogrulandi = (dogrula_sayisal_tutarlilik(cevap, web_context)
                  and dogrula_eksik_deger(cevap, web_context))
    if not dogrulandi:
        cevap = _dogrulanmamis_fallback(web_context, "web")

    return cevap, "web", dogrulandi
