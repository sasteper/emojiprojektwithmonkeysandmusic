
# 🎭 Emoji Face Tracker (Python)

Aplikacja w Pythonie działająca w czasie rzeczywistym, która rozpoznaje mimikę twarzy i gesty z kamery oraz wyświetla odpowiadające im emoji z muzyką w tle.

---

## 📌 Opis projektu

**Emoji Face Tracker** to interaktywny projekt w Pythonie analizujący obraz z kamery w czasie rzeczywistym.  
Na podstawie mimiki twarzy i prostych gestów aplikacja wyświetla odpowiednie emoji oraz odtwarza muzykę (WAV) w pętli.

Projekt działa **lokalnie**, bez połączenia z internetem.

---

## 🎯 Funkcje

- Rozpoznawanie mimiki twarzy i gestów
- Reakcje emoji w czasie rzeczywistym
- Obsługa jednej osoby przed kamerą
- Odtwarzanie muzyki w pętli
- Sterowanie klawiaturą

### Wykrywane stany:
- `HANDS_UP` – uniesione ręce  
- `SMILING` – uśmiech  
- `CLOSED_EYES` – zamknięte oczy  
- `SHOCKED` – szeroko otwarte usta  
- `ANGRY` – zmarszczone brwi  
- `THINKING` – palec blisko ust  
- `CURIOUS` – głowa obrócona w bok  
- `TONGUE` – widoczny język  
- `STARE` – stan neutralny  

---

## 🛠 Wymagania

### System
- Windows / macOS / Linux
- Kamera internetowa

### Python
- **Python 3.9 – 3.11 (zalecane)**
- Python 3.12 zwykle działa, ale w razie problemów zalecany jest Python 3.11

Sprawdzenie wersji:
```bash
python --version
