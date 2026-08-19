# -*- coding: utf-8 -*-
"""
Qwen2.5-7B-Instruct modelini (GGUF / q4_k_m) Hugging Face'ten indirir.

download_model.py ile aynı mantık: huggingface_hub'un hf_hub_download
fonksiyonu ile models/ klasörüne çeker. 7B q4_k_m quantizasyonu resmi
Qwen deposunda İKİ shard halinde tutulur (00001-of-00002 / 00002-of-00002);
bu yüzden iki dosya da indirilir. llama.cpp ilk shard verildiğinde diğerini
otomatik bulur, bu nedenle model yolu ilk shard'ı gösterir.
Toplam boyut yaklaşık ~4.4 GB; indirme internet hızına göre uzun sürebilir.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

# 7B modelinin Hugging Face'teki deposu ve q4_k_m shard dosyaları
REPO_ID = "Qwen/Qwen2.5-7B-Instruct-GGUF"
FILENAMES = [
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
]

# Bu script src/ klasörünün içinde; "bir üst klasördeki models/" hedef klasör.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Depo  : {REPO_ID}")
    print(f"Hedef : {MODELS_DIR}")
    print(f"{len(FILENAMES)} shard indirilecek (toplam ~4.4 GB)...\n")

    for filename in FILENAMES:
        print(f"İndiriliyor: {filename}")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=MODELS_DIR,
        )
        print(f"  -> {local_path}\n")

    print("Tüm shardlar indirildi.")


if __name__ == "__main__":
    main()
