# -*- coding: utf-8 -*-
"""
Source filtrelemesinin testi.

retrieval_araci artık top_k chunk'ları source bazında gruplayıp çoğunluk source'una
indirger ve (context, source) döndürür. Bu script iki soruyu çalıştırıp şunları doğrular:

1. Filtreleme ÖNCESİ top_k'da kaç farklı source olduğu (ham dağılım),
2. Filtreleme SONRASI hangi tek source'un kaldığı (retrieval_araci dönüşü),
3. Nihai fallback mesajının doğru dosyadan ("Kaynak: ...") gelip gelmediği.
"""

from collections import Counter

from llama_cpp import Llama

from rag import (
    DEFAULT_MESAFE_ESIGI,
    MODEL_PATH,
    _get_collection,
    _get_embedding_model,
    generate_answer,
    load_embedding_model,
    retrieval_araci,
)

# Soru -> (etiket, beklenen doğru source dosyası)
SORULAR = [
    ("BilgiTR projesinde hit-rate sonucu nedir?", "bilgitr-proje.txt"),
    ("HizmetGelsin'in backend'i hangi teknolojilerle yazıldı?", "hizmetgelsin-proje.txt"),
    ("YorumTR projesinde hangi yöntem daha iyi sonuç verdi?", "yorumtr-proje.txt"),
]

TOP_K = 4


def _ham_source_dagilimi(soru: str) -> list[str]:
    """Source filtrelemesinden ÖNCE (mesafe eşiğini geçen) chunk'ların source listesi.

    retrieval_araci'nin çoğunluk oyu öncesinde neyle çalıştığını görmek için aynı
    adımları tekrar ederiz: top_k sorgusu + mesafe eşiği filtrelemesi (source
    gruplaması HARİÇ). Bu, eski davranışın bağlama karıştırdığı chunk'ları gösterir.
    """
    model = _get_embedding_model()
    collection = _get_collection()
    qe = model.encode([soru]).tolist()
    res = collection.query(query_embeddings=qe, n_results=TOP_K)
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    return [
        (m or {}).get("source", "")
        for _doc, dist, m in zip(docs, dists, metadatas)
        if dist <= DEFAULT_MESAFE_ESIGI
    ]


def _kisa_dagilim(sources: list[str]) -> str:
    """Source listesini "ad (n)" biçiminde kısa bir özete çevirir."""
    sayim = Counter(sources)
    return ", ".join(f"{s or '(bos)'} ({n})" for s, n in sayim.most_common())


def main() -> None:
    print("Embedding + 3B model yükleniyor...")
    load_embedding_model()
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=3072, n_threads=4, verbose=False)
    print("Model hazır.\n")

    tum_ok = True
    for i, (soru, beklenen_source) in enumerate(SORULAR, start=1):
        ham = _ham_source_dagilimi(soru)
        context, source = retrieval_araci(soru)

        print("=" * 72)
        print(f"S{i}: {soru}")
        print(f"  [ÖNCE] top_k={TOP_K} ham source dağılımı: {_kisa_dagilim(ham)} "
              f"({len(set(ham))} farklı source)")
        print(f"  [SONRA] seçilen çoğunluk source   : {source or '(boş)'} "
              f"({len(set(ham)) - 1 if len(set(ham)) > 1 else 0} yabancı source elendi)")

        cevap, kaynak, dogrulandi = generate_answer(llm, soru)
        fallback = cevap.startswith("Modelin ürettiği cevap doğrulanamadı")

        # Ana doğrulama: retrieval doğru source'u seçti mi?
        dogru_source = (source == beklenen_source)
        # Fallback olduysa mesaj da doğru dosyayı içermeli; fallback yoksa ayrıca
        # kontrol edilecek bir şey yok (model cevabı kullanılmış demektir).
        dogru_fallback = (beklenen_source in cevap) if fallback else True

        print(f"  generate_answer: kaynak={kaynak} | doğrulandı={dogrulandi} | "
              f"fallback={'EVET' if fallback else 'HAYIR'}")
        print(f"  [retrieval] doğru source seçildi mi ('{beklenen_source}')? "
              f"{'EVET' if dogru_source else 'HAYIR'}")
        if fallback:
            print(f"  [fallback] mesaj doğru dosyayı içeriyor mu? "
                  f"{'EVET' if dogru_fallback else 'HAYIR'}")

        ok = dogru_source and dogru_fallback
        tum_ok = tum_ok and ok
        print(f"  --> {'GEÇTİ' if ok else 'BAŞARISIZ'}")
        if fallback:
            print("  nihai cevap (ilk 120 kr):")
            print("    " + cevap[:120].replace("\n", "\n    "))
        print()

    print("=" * 72)
    print("SONUÇ:", "TÜMÜ GEÇTİ" if tum_ok else "BAZILARI BAŞARISIZ")
    raise SystemExit(0 if tum_ok else 1)


if __name__ == "__main__":
    main()
