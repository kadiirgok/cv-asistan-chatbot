# -*- coding: utf-8 -*-
"""
Agent katmanının uçtan uca testi (Adım 2b).

agent_yanit() üzerinden 3 senaryo denenir ve her birinde modelin hangi aracı
seçtiği, seçimin beklenene uyup uymadığı raporlanır:

  1. Basit genel bilgi sorusu  -> hiç araç çağırmamalı (None).
  2. Local DB'deki bir konu   -> retrieval_araci çağırmalı.
  3. Güncel/web gerektiren    -> web_search_araci çağırmalı.

Model, agent.py içinde ilk çağrıda bir kez yüklenir ve sonraki senaryolarda
aynı nesne kullanılır (yeniden yüklenmez).
"""

from agent import agent_yanit

# (başlık, soru, beklenen araç) üçlüleri. "beklenen_arac" None ise modelin
# hiç araç çağırmaması beklenir.
SENARYOLAR = [
    {
        "baslik": "Basit genel bilgi (araçsız beklenir)",
        "soru": "2 + 2 kaç eder?",
        "beklenen_arac": None,
    },
    {
        "baslik": "Local DB konusu (retrieval beklenir)",
        "soru": "Mercimek çorbası nasıl yapılır?",
        "beklenen_arac": "retrieval_araci",
    },
    {
        "baslik": "Güncel/web gerektiren (web beklenir)",
        "soru": "Güncel dolar kuru kaç TL?",
        "beklenen_arac": "web_search_araci",
    },
]


def main():
    print("=" * 72)
    print("AGENT TESTİ — 3 senaryo")
    print("=" * 72)

    for i, sen in enumerate(SENARYOLAR, 1):
        print(f"\n### [{i}/{len(SENARYOLAR)}] {sen['baslik']}")
        print(f"SORU: {sen['soru']}")
        print("-" * 72)

        sonuc = agent_yanit(sen["soru"])

        print(f"Seçilen araç : {sonuc['kullanilan_arac']}")
        print(f"Süre         : {sonuc['sure_saniye']:.2f} sn")
        print("CEVAP:")
        print(sonuc["cevap"])

        # Araç seçimini beklenenle karşılaştır
        secilen = sonuc["kullanilan_arac"]
        beklenen = sen["beklenen_arac"]
        if secilen == beklenen:
            print(f"\n[DOĞRU] Beklenen araç ({beklenen}) seçildi.")
        else:
            print(f"\n[YANLIŞ] Beklenen: {beklenen}, seçilen: {secilen}")
        print("=" * 72)

    print("\nTest tamamlandı.")


if __name__ == "__main__":
    main()
