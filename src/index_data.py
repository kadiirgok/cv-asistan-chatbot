# -*- coding: utf-8 -*-
"""
data/ klasöründeki .txt dosyalarını okuyup ChromaDB'ye indeksleyen script.

Kullanım: data/ klasörüne .txt dosyalarını bırak, sonra bu scripti çalıştır.
(seed_test_data.py test verisi için hâlâ duruyor; bu script üretim indekslemesi içindir.)
"""

from rag import build_index, load_documents_from_folder


def main():
    """data/ klasöründeki dokümanları okur ve build_index ile indeksler."""
    documents = load_documents_from_folder("./data")

    if not documents:
        print("data/ klasöründe indekslenecek .txt dosyası bulunamadı.")
        print("İndeks değiştirilmedi (mevcut veri korundu).")
        return

    print(f"{len(documents)} doküman bulundu:\n")
    for doc in documents:
        print(f"  - {doc['source']} ({len(doc['text'])} karakter)")

    print()
    build_index(documents)


if __name__ == "__main__":
    main()
