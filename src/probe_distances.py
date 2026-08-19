# -*- coding: utf-8 -*-
"""
Tanılama scripti: seçili embedding modeliyle anahtar soruların en yakın chunk
mesafelerini (distance) ve embedding hızını yazdırır.

Model ve persist klasörü, rag.py ile aynı ortam değişkenlerinden okunur
(EMBEDDING_MODEL, CHROMA_DB_DIR). Örnek kullanım:
  EMBEDDING_MODEL="..." CHROMA_DB_DIR="chroma_db_x" python src/probe_distances.py
"""

import time

import chromadb

from rag import COLLECTION_NAME, EMBEDDING_MODEL_NAME, PERSIST_DIR, _get_embedding_model

# Mesafelerine bakılacak anahtar sorular (stress testteki sorularla aynı)
SORULAR = [
    ("a) deprem/Richter", "Deprem büyüklüğünü ölçmek için hangi ölçek kullanılır?"),
    ("d1) Ege/İzmir",     "Ege Bölgesi'nde yer alan şehir hakkında bilgi ver."),
    ("d2) Marmara/Bursa", "Marmara Bölgesi'nde yer alan şehir hakkında bilgi ver."),
    ("b) Bursa",          "Bursa'nın tarihi önemini ve ekonomisinin temel sektörlerini anlat."),
    ("c) Ay (alakasız)",  "Ay'a ilk insanlı iniş hangi yılda gerçekleşti?"),
    ("mercimek",          "Mercimek çorbası nasıl yapılır?"),
]


def main():
    print(f"Model  : {EMBEDDING_MODEL_NAME}")
    print(f"Persist: {PERSIST_DIR}\n")

    model = _get_embedding_model()
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    # --- Embedding hızı ölçümü (sabit bir metin grubu) ---
    ornek = [
        "Deprem büyüklüğünü ölçmek için Richter ölçeği kullanılır.",
        "Bursa, Marmara Bölgesi'nin güneyinde yer alan tarihi bir şehirdir.",
        "Mercimek çorbası kırmızı mercimek, soğan ve havuçla yapılır.",
        "Mikroservis mimarisi bağımsız servislerden oluşur.",
        "Çanakkale Savaşı 1915 yılında gerçekleşmiştir.",
    ] * 20  # 100 kısa metin
    t0 = time.time()
    model.encode(ornek, show_progress_bar=False)
    embed_sure = time.time() - t0
    print(f"[Hız] 100 kısa metin embed: {embed_sure:.2f} sn\n")

    # --- Mesafe değerleri ---
    for baslik, soru in SORULAR:
        emb = model.encode([soru]).tolist()
        res = col.query(query_embeddings=emb, n_results=5)
        dists = res["distances"][0]
        docs = res["documents"][0]
        print(f"=== {baslik} ===")
        print(f"  Soru: {soru}")
        for i, (d, doc) in enumerate(zip(dists, docs)):
            snippet = " ".join(doc.split()[:7])
            print(f"    [{i+1}] dist={d:.4f}  -> {snippet}...")
        print()


if __name__ == "__main__":
    main()
