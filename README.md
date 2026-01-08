
# 🎭 Emoji Face Tracker z muzyką (Python)

Aplikacja w Pythonie, która w czasie rzeczywistym analizuje obraz z kamery, rozpoznaje mimikę twarzy oraz proste gesty i wyświetla odpowiadające im emoji.  
Dodatkowo program odtwarza muzykę w formacie WAV w pętli pokazuje tekst.

Głównym celem artystycznym tego projektu jest pokazanie, że jesteśmy jednocześnie milionem różnych ludzi (małp) i że nieustannie się zmieniamy.

---

## 📌 Opis projektu

Emoji Face Tracker to interaktywny projekt wykorzystujący OpenCV (kamera/okna) oraz MediaPipe (detekcja twarzy i dłoni).  
W czasie rzeczywistym aplikacja rozpoznaje wybrane stany mimiczne oraz gesty użytkownika i wyświetla odpowiednie emoji — równocześnie w tle odtwarzając muzykę (WAV) w pętli.

Projekt można użyć jako:
- demonstrację Computer Vision / projekt zaliczeniowy,
- element streamingu (OBS),
- interaktywną instalację lub mini‑performance.

---

## 🎯 Funkcje

- Rozpoznawanie mimiki twarzy i prostych gestów z kamery.
- Reakcje emoji w czasie rzeczywistym.
- Odtwarzanie muzyki (WAV) w pętli.
- Proste sterowanie muzyką z klawiatury.
- Działanie lokalne, bez wysyłania danych i bez zapisu wideo.

### Rozpoznawane stany:

- `HANDS_UP` – uniesione ręce  
- `SMILING` – uśmiech  
- `CLOSED_EYES` – zamknięte oczy  
- `SHOCKED` – szeroko otwarte usta  
- `ANGRY` – złość (zmarszczone brwi)  
- `THINKING` – palec blisko ust  
- `CURIOUS` – głowa obrócona w bok  
- `TONGUE` – widoczny język  
- `STARE` – stan neutralny (domyślny)

---

## 🛠 Wymagania

### System
- Windows / macOS / Linux
- Kamera internetowa

### Python
- Python 3.9 – 3.11 (zalecane)
- Python 3.12 zwykle działa; jeśli pojawią się problemy z `mediapipe`, użyj Python 3.11.

Sprawdzenie wersji:
```bash
python --version


📁 Struktura projektu (pliki wymagane)
project/
 ├─ app.py                       # główny plik programu
 ├─ bitter sweet symphony.wav    # muzyka (WAV, odtwarzana w pętli)
 └─ assets/                      # folder z grafikami emoji
    ├─ air.jpg
    ├─ evil_smile.jpeg
    ├─ closed_eyes.jpeg
    ├─ staring.jpeg
    ├─ shocked_monki.jpeg
    ├─ angry_monki.jpeg
    ├─ thinking_monki.jpeg
    ├─ curious_monki.jpeg
    └─ tongue.jpeg

Ważne:

plik muzyczny musi być WAV (np. bitter sweet symphony.wav),
obrazki mogą być .jpg lub .jpeg,
brak pojedynczej grafiki nie wywołuje błędu; po prostu dana reakcja nie będzie się wyświetlać.


▶️ Uruchomienie

Otwórz terminal w folderze projektu.
Uruchom:

Shellpython app.pyShow more lines
Po starcie pojawią się dwa okna:

Camera — obraz z kamery,
Emoji — aktualna reakcja emoji.

W tle uruchomi się muzyka (WAV) w pętli.



🔧 Technologie

Python
OpenCV — obsługa kamery i okien
MediaPipe — detekcja twarzy, oczu, ust, dłoni
NumPy + HSV — wykrywanie języka na podstawie koloru w ROI ust
pygame.mixer — odtwarzanie muzyki WAV w pętli




⚠️ Rozwiązywanie problemów (skrót)

Kamera: upewnij się, że działa w innych aplikacjach; na macOS zaakceptuj dostęp do kamery.
Python: najlepsza kompatybilność z 3.10 / 3.11.
Muzyka: plik musi być WAV (MP3 nie zadziała z bieżącą konfiguracją).
Biblioteki: jeśli mediapipe ma problem z instalacją na 3.12, użyj 3.11.


✅ W streszczeniu: Zainstaluj Python 3.10/3.11, doinstaluj wymagane biblioteki, upewnij się, że pliki (app.py, muzyka WAV, folder assets/) są na miejscu i uruchom python app.py — zaakceptuj dostęp do kamery

