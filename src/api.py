# -*- coding: utf-8 -*-
"""
FastAPI HTTP katmanı.

Bu dosya yalnızca HTTP ile ilgili işleri içerir: endpoint tanımları, CORS ve
model yükleme. İş mantığı (retrieval + prompt + cevap üretimi) rag.py içindeki
generate_answer fonksiyonunda tutulur.
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llama_cpp import Llama

# `uvicorn src.api:app` ile çalıştırıldığında src/ klasörü sys.path'te bulunmaz;
# aynı klasördeki rag modülünü içe aktarabilmek için src/ yolunu en başa ekle.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import MODEL_PATH, generate_answer, load_embedding_model

# Model global değişkende tutulur (her istekte yeniden yüklenmez)
llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlarken modeli ve embedding modelini bir kez yükler."""
    global llm
    print("Model yükleniyor...")
    llm = Llama(
        model_path=str(MODEL_PATH),  # model dosyasının yolu
        n_ctx=3072,                  # bağlam penceresi (3B ile test edildi, kaliteyi korur)
        n_threads=2,                 # kullanılacak CPU çekirdeği sayısı (2: bellek baskısını azaltır)
        verbose=False,               # yükleme loglarını kapat
    )
    load_embedding_model()  # embedding modelini de önceden yükle (ilk isteği hızlandırır)
    print("Model hazır.")
    yield
    # (Uygulama kapanırken istenirse temizlik yapılabilir)


app = FastAPI(title="Chatbot RAG API", lifespan=lifespan)

# CORS: ileride Flutter gibi farklı origin'lerden erişebilmek için herkese açık bırak
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SoruRequest(BaseModel):
    """POST /chat isteğinin gövdesi."""
    soru: str


class CevapResponse(BaseModel):
    """POST /chat yanıtının gövdesi."""
    cevap: str
    kaynak: str
    sure_saniye: float
    dogrulandi: bool


@app.get("/health")
def health():
    """Sistemin ayakta olup olmadığını kontrol eden basit uç nokta."""
    return {"durum": "hazir"}


@app.post("/chat", response_model=CevapResponse)
def chat(req: SoruRequest):
    """Soruyu RAG ile yanıtlar; cevabı ve geçen süreyi döndürür."""
    t0 = time.time()
    cevap, kaynak, dogrulandi = generate_answer(llm, req.soru)  # iş mantığı rag.py'de
    sure = time.time() - t0
    return CevapResponse(cevap=cevap, kaynak=kaynak, sure_saniye=round(sure, 3), dogrulandi=dogrulandi)


@app.exception_handler(Exception)
async def genel_hata_handler(request: Request, exc: Exception):
    """Beklenmeyen bir istisna olursa isteği çökertmek yerine nazik bir JSON yanıtı döner.

    Not: Bu, Python seviyesindeki istisnaları yakalar; llama.cpp'nin yerel (C++) katmanındaki
    bir segmentation fault sürecin kendisini çökertir ve bu handler'a düşmez.
    """
    return JSONResponse(
        status_code=500,
        content={
            "cevap": "Sunucu beklenmeyen bir hata oluştu. Lütfen tekrar deneyin.",
            "kaynak": "none",
            "sure_saniye": 0.0,
            "dogrulandi": False,
        },
    )


# --- Statik arayüz ---
# Aynı FastAPI uygulaması hem API'yi hem de arayüzü sunar (ayrı bir web sunucusu
# gerekmez). static/index.html tek dosyalık sohbet arayüzüdür; "/" adresi onu açar.
# Mount EN SONA eklenir ki /chat ve /health gibi rotalar önce eşleşsin.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
