# CV Asistanı (Flutter)

Backend chatbot'a (`chatbot-projesi`) bağlanan basit bir Flutter sohbet arayüzü.
Tek ekran, `http` paketi ile `/chat` uç noktasına POST isteği atar; mesajları
balonlar halinde listeler, her asistan cevabının altında kaynağını ve doğrulama
durumunu gösterir.

## Gereksinimler

- Flutter SDK kurulu olmalı.
- **Backend önce ayakta olmalı.** Backend'i şu şekilde başlat:

  ```bash
  cd chatbot-projesi
  venv/Scripts/python.exe -m uvicorn src.api:app
  ```

## Çalıştırma

```bash
flutter pub get
flutter run
```

## API adresi (önemli)

`lib/api_service.dart` içindeki `apiBaseUrl` şu an Android emülatörü için
`http://10.0.2.2:8000` olarak ayarlı. `10.0.2.2`, Android emülatöründen host
makinenin `localhost`'una erişim için kullanılan özel adrestir.

- **Android emülatörü / BlueStacks:** `10.0.2.2` genelde doğrudan çalışır.
  (BlueStacks bu adrese `adb reverse` ya da özel yapılandırma gerektirebilir.)
- **Gerçek cihaz:** `apiBaseUrl`'i bilgisayarın ağ IP'si ile değiştir, örneğin
  `http://192.168.1.20:8000`. Telefon ile bilgisayar aynı ağda olmalı ve backend
  ağdan erişilebilir olmalı:

  ```bash
  uvicorn src.api:app --host 0.0.0.0 --port 8000
  ```

## Hata yönetimi

Sunucuya ulaşılamazsa (backend kapalı, ağ hatası, yanlış IP) uygulama çökmez;
sohbet akışında "Sunucuya ulaşılamadı. Backend'in çalıştığından emin olun."
benzeri bir hata mesajı gösterir.

## Derleme (APK)

```bash
flutter build apk --debug
```

Çıktı: `build/app/outputs/flutter-apk/app-debug.apk`
