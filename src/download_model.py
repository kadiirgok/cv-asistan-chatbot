# -*- coding: utf-8 -*-
"""
Qwen2.5-1.5B-Instruct modelini (GGUF / q4_k_m) Hugging Face'ten indirir.
Tarayıcı kullanmadan, huggingface_hub kütüphanesinin hf_hub_download
fonksiyonu ile tek dosyayı models/ klasörüne çeker.
"""

# Yolları taşınabilir şekilde ele almak için pathlib kullanıyoruz
from pathlib import Path

# Tek bir model dosyasını indiren fonksiyon
from huggingface_hub import hf_hub_download

# Modelin Hugging Face'teki deposu (repo id) ve istediğimiz dosyanın adı
REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Bu script src/ klasörünün içinde; "bir üst klasördeki models/" hedef klasör.
# (__file__ = bu dosyanın yolu, .parent.parent = src'in bir üstü = proje kökü)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    # models/ klasörü yoksa oluştur (var olanı bozmaz)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Depo  : {REPO_ID}")
    print(f"Dosya : {FILENAME}")
    print(f"Hedef : {MODELS_DIR}")
    print("İndirme başlıyor (yaklaşık ~1 GB, internet hızına göre sürebilir)...\n")

    # Belirtilen tek dosyayı models/ klasörüne indirir
    local_path = hf_hub_download(
        repo_id=REPO_ID,     # hangi repo
        filename=FILENAME,   # repodaki hangi dosya
        local_dir=MODELS_DIR,  # nereye kaydedilecek
    )

    print(f"\nTamamlandı: {local_path}")


if __name__ == "__main__":
    main()
