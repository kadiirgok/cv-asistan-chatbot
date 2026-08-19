# -*- coding: utf-8 -*-
"""
RAG'i test etmek için birbirinden bağımsız Türkçe belgeleri oluşturup
ChromaDB'ye indeksleyen script. Sohbet testinden ÖNCE bu script çalıştırılmalıdır.
"""

from rag import build_index


def main():
    """Birbirinden bağımsız 6 Türkçe test dokümanını indeksler."""
    belgeler = [
        # 1) Bir şehir hakkında bilgi
        "İzmir, Ege Bölgesi'nde yer alan Türkiye'nin üçüncü büyük şehridir. "
        "Körfez kıyısında kurulmuştur ve ılıman Akdeniz iklimine sahiptir. "
        "Tarihi Saat Kulesi ve Kordon boyu şehrin simgeleri arasındadır.",

        # 2) Bir yemek tarifi
        "Mercimek çorbası; kırmızı mercimek, soğan, havuç ve salça ile yapılan "
        "geleneksel bir Türk çorbasıdır. Malzemeler piştikten sonra blenderdan "
        "geçirilir ve üzerine limon sıkılarak servis edilir.",

        # 3) Bir teknoloji açıklaması
        "Yapay zeka, bilgisayarların öğrenme, akıl yürütme ve karar verme gibi "
        "insan benzeri yetenekleri taklit etmesini sağlayan bir teknolojidir. "
        "Makine öğrenmesi ise bu alanın verilerden örüntü çıkaran alt dalıdır.",

        # 4) Bir tarih bilgisi
        "İstanbul'un fethi 1453 yılında Fatih Sultan Mehmet önderliğinde "
        "gerçekleşmiştir. Bu fetih Bizans İmparatorluğu'nun sonunu getirmiş ve "
        "İstanbul, Osmanlı Devleti'nin başkenti olmuştur.",

        # 5) Bir bilim/doğa açıklaması
        "Fotosentez, bitkilerin güneş ışığını kullanarak karbondioksit ve sudan "
        "glikoz üretmesini sağlayan kimyasal bir süreçtir. Bu süreç sırasında "
        "atmosfere oksijen salınır.",

        # 6) Bir spor açıklaması
        "Basketbol, beşer kişilik iki takımın bir topu karşı takımın potasına "
        "atarak sayı kazanmaya çalıştığı bir spordur. Maçlar dört periyottan oluşur "
        "ve her isabetli atış iki veya üç sayı değerindedir.",
    ]

    print(f"{len(belgeler)} test dokümanı indeksleniyor...\n")
    build_index(belgeler)


if __name__ == "__main__":
    main()
