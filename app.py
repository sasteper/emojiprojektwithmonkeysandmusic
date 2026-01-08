
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emoji + WAV music + Lyrics (typewriter 1.9 сек на рядок, looping)

Music: bitter sweet symphony.wav
Lyrics: lyrics.txt (UTF-8)
"""

import cv2
import numpy as np
import math
import time
from pathlib import Path
from collections import deque

# ========= MediaPipe =========
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# ========= Window / Emoji =========
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# ========= Thresholds =========
SMILE_MAR = 0.36
SMILE_MAR_MAX = 0.52
SMILE_EYE_MAX = 0.58
SURPRISE_MAR = 0.62
EYE_CLOSE_EAR = 0.42
ANGRY_BROW_FACTOR = 0.80
FINGER_MOUTH_DIST = 0.060
YAW_SIDE_FACE = 0.18
TONGUE_MAR_MIN = 0.38
TONGUE_COLOR_MIN_RATIO = 0.08
TONGUE_ROI_BOTTOM_FRACTION = 0.55
TONGUE_MAX_EYE_WIDE = 0.595

# ========= Paths =========
SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"
def asset_path(name: str):
    p1 = SCRIPT_DIR / name
    p2 = ASSETS_DIR / name
    return p1 if p1.exists() else p2

# ========= Lyrics =========
TEXT_WIN_WIDTH  = 600
TEXT_WIN_HEIGHT = 220
TEXT_BG_COLOR   = (0, 206, 138)
TEXT_FG_COLOR   = (230, 230, 230)
TEXT_FONT       = cv2.FONT_HERSHEY_SIMPLEX
TEXT_FONT_SCALE = 0.7
TEXT_THICKNESS  = 1
LINE_TYPE_DURATION_SEC = 1.9
MAX_VISIBLE_LINES = 6
LYRICS_FILE = "lyrics.txt"

def load_lyrics_lines():
    p = asset_path(LYRICS_FILE)
    if p.exists():
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        print("[lyrics] file not found:", p)
        return []

def render_lyrics(canvas, lines_buffer, current_line, chars_to_show):
    canvas[:] = TEXT_BG_COLOR
    x, y = 10, 30
    recent = list(lines_buffer)[-MAX_VISIBLE_LINES:]
    for ln in recent:
        cv2.putText(canvas, ln, (x, y), TEXT_FONT, TEXT_FONT_SCALE, TEXT_FG_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
        y += 28
    if current_line:
        cv2.putText(canvas, current_line[:chars_to_show], (x, y), TEXT_FONT, TEXT_FONT_SCALE, TEXT_FG_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
    return canvas

# ========= Emoji helpers =========
def imread_smart(filename):
    img = cv2.imread(str(SCRIPT_DIR / filename), cv2.IMREAD_UNCHANGED)
    if img is None:
        img = cv2.imread(str(ASSETS_DIR / filename), cv2.IMREAD_UNCHANGED)
    return img

def resize(img):
    return cv2.resize(img, EMOJI_WINDOW_SIZE, interpolation=cv2.INTER_AREA) if img is not None else None

# ========= Geometry helpers (FaceMesh) =========
def _lm_xy(face, idx):
    # MediaPipe FaceMesh landmarks are normalized (0..1)
    lm = face.landmark[idx]
    return lm.x, lm.y

def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def face_width(face):
    xs = [lm.x for lm in face.landmark]
    return (max(xs) - min(xs)) if xs else 1e-6

def eye_aspect_ratio(face, top_idx, bottom_idx, left_idx, right_idx):
    top = _lm_xy(face, top_idx)
    bottom = _lm_xy(face, bottom_idx)
    left = _lm_xy(face, left_idx)
    right = _lm_xy(face, right_idx)
    v = _dist(top, bottom)
    h = _dist(left, right)
    return (v / h) if h > 1e-6 else 0.0

def mouth_metrics(face):
    """
    Простий MAR: вертикаль (центр верхньої/нижньої губи) / горизонталь (кути рота).
    Індекси взяті з типової FaceMesh:
      - 13: центр верхньої внутрішньої губи
      - 14: центр нижньої внутрішньої губи
      - 61: лівий кут рота
      - 291: правий кут рота
    """
    try:
        up = _lm_xy(face, 13)
        down = _lm_xy(face, 14)
        left = _lm_xy(face, 61)
        right = _lm_xy(face, 291)
    except Exception:
        return 0.0
    v = _dist(up, down)
    h = _dist(left, right)
    return (v / h) if h > 1e-6 else 0.0

def brow_eye_dist(face):
    """
    Оцінка "насупленості": середня відстань між бровою та верхом ока (ліва/права),
    нормалізована на ширину обличчя.
    Індекси (наближено):
      - Ліва брова: 70, верх ока: 386
      - Права брова: 300, верх ока: 159
    """
    fw = face_width(face)
    try:
        lb = _lm_xy(face, 70); le = _lm_xy(face, 386)
        rb = _lm_xy(face, 300); re = _lm_xy(face, 159)
    except Exception:
        return 0.0
    d1 = _dist(lb, le)
    d2 = _dist(rb, re)
    mean = (d1 + d2) / 2.0
    return (mean / fw) if fw > 1e-6 else 0.0

def head_yaw_ratio(face):
    """
    Дуже проста оцінка повороту голови: різниця відстаней від носа до лівої/правої щоки,
    нормалізована на ширину обличчя.
    Індекси (орієнтовно):
      - Ніс: 1
      - Ліва щока/скула: 234
      - Права щока/скула: 454
    """
    fw = face_width(face)
    try:
        nose = _lm_xy(face, 1)
        left = _lm_xy(face, 234)
        right = _lm_xy(face, 454)
    except Exception:
        return 0.0
    dl = _dist(nose, left)
    dr = _dist(nose, right)
    return ((dr - dl) / fw) if fw > 1e-6 else 0.0

def finger_in_mouth_any(hand_landmarks, face, fw):
    """
    Стаб: повертає False, щоб не падало. За потреби додамо перевірку
    близькості кінчиків пальців до рота.
    """
    return False

def tongue_color_ratio_in_mouth(face, frame_bgr):
    """
    Стаб: повертає 0.0. За потреби можна зробити HSV-детекцію червоно-рожевих пікселів
    у ROI рота.
    """
    return 0.0

# ========= Camera =========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera not available")

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Emoji", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow("Emoji", WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow("Camera", 100, 100)
cv2.moveWindow("Emoji", 100+WINDOW_WIDTH+30, 100)

# ========= Load emojis =========
hands_up_emoji    = resize(imread_smart("air.jpg"))
evil_smile_emoji  = resize(imread_smart("evil_smile.jpeg"))
closed_eyes_emoji = resize(imread_smart("closed_eyes.jpeg"))
staring_emoji     = resize(imread_smart("staring.jpeg"))
shocked_emoji     = resize(imread_smart("shocked_monki.jpeg"))
angry_emoji       = resize(imread_smart("angry_monki.jpeg"))
thinking_emoji    = resize(imread_smart("thinking_monki.jpeg"))
curious_emoji     = resize(imread_smart("curious_monki.jpeg"))
tongue_emoji      = resize(imread_smart("tongue.jpeg"))
blank = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

# ========= Load lyrics =========
lyrics_lines = load_lyrics_lines()
lyrics_queue = deque(lyrics_lines)
lines_buffer = deque()
current_line = ""
current_start_time = None

text_canvas = np.zeros((TEXT_WIN_HEIGHT, TEXT_WIN_WIDTH, 3), dtype=np.uint8)
cv2.namedWindow("Lyrics", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Lyrics", TEXT_WIN_WIDTH, TEXT_WIN_HEIGHT)
cv2.moveWindow("Lyrics", 100+WINDOW_WIDTH+30, 100+WINDOW_HEIGHT+40)

# ========= Music WAV =========
pygame_available = False
try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 2048)
    pygame.mixer.init()
    music_path = asset_path("bitter sweet symphony.wav")
    if music_path.exists():
        pygame.mixer.music.load(str(music_path))
        pygame.mixer.music.set_volume(0.85)
        pygame.mixer.music.play(loops=-1)
        pygame_available = True
        print("[music] playing loop")
    else:
        print("[music] file not found:", music_path)
except Exception as e:
    print("[music] failed:", e)

# ========= Main loop =========
with mp_pose.Pose() as pose, mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh, mp_hands.Hands(max_num_hands=1) as hands:
    baseline_brow = None
    brow_samples = []
    baseline_mar = None
    mar_samples = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_r = pose.process(rgb)
        face_r = face_mesh.process(rgb)
        hand_r = hands.process(rgb)

        # ===== Emoji detection (simplified) =====
        hands_up = False
        shocked = closed = angry = thinking = smile = False
        curious = tongue = False

        # Pose: руки підняті
        if pose_r.pose_landmarks:
            lm = pose_r.pose_landmarks.landmark
            try:
                if (lm[mp_pose.PoseLandmark.LEFT_WRIST].y < lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y) or \
                   (lm[mp_pose.PoseLandmark.RIGHT_WRIST].y < lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y):
                    hands_up = True
            except Exception:
                hands_up = False

        if face_r.multi_face_landmarks:
            face = face_r.multi_face_landmarks[0]
            mar = mouth_metrics(face)
            ear_l = eye_aspect_ratio(face, 386, 374, 263, 362)  # ліва
            ear_r = eye_aspect_ratio(face, 159, 145, 33, 133)   # права
            brow = brow_eye_dist(face)
            yaw = head_yaw_ratio(face)
            fw = face_width(face)

            # baseline
            if baseline_brow is None:
                brow_samples.append(brow)
                if len(brow_samples) >= 20:
                    baseline_brow = float(np.mean(brow_samples))
            if baseline_mar is None:
                mar_samples.append(mar)
                if len(mar_samples) >= 20:
                    baseline_mar = float(np.median(mar_samples))

            if mar > SURPRISE_MAR:
                shocked = True
            elif (ear_l < EYE_CLOSE_EAR) and (ear_r < EYE_CLOSE_EAR):
                closed = True
            elif (baseline_brow is not None) and (brow < baseline_brow * ANGRY_BROW_FACTOR) and (mar < 0.45):
                angry = True

            if abs(yaw) >= YAW_SIDE_FACE:
                curious = True

            if hand_r.multi_hand_landmarks:
                try:
                    if finger_in_mouth_any(hand_r.multi_hand_landmarks[0], face, fw):
                        thinking = True
                except Exception:
                    thinking = False

            eyes_not_wide = (ear_l <= TONGUE_MAX_EYE_WIDE and ear_r <= TONGUE_MAX_EYE_WIDE)
            if (mar >= max(TONGUE_MAR_MIN, (baseline_mar or 0.0) + 0.06)) and eyes_not_wide and not shocked:
                ratio = tongue_color_ratio_in_mouth(face, frame)
                if ratio >= TONGUE_COLOR_MIN_RATIO:
                    tongue = True

            if (SMILE_MAR <= mar <= SMILE_MAR_MAX) and (not tongue) and (not shocked) and (not angry) and \
               (ear_l < SMILE_EYE_MAX) and (ear_r < SMILE_EYE_MAX):
                smile = True

        # Priority
        if hands_up and (hands_up_emoji is not None):
            emoji, state = hands_up_emoji, "HANDS_UP"
        elif shocked and (shocked_emoji is not None):
            emoji, state = shocked_emoji, "SHOCKED"
        elif angry and (angry_emoji is not None):
            emoji, state = angry_emoji, "ANGRY"
        elif closed and (closed_eyes_emoji is not None):
            emoji, state = closed_eyes_emoji, "CLOSED_EYES"
        elif curious and (curious_emoji is not None):
            emoji, state = curious_emoji, "CURIOUS"
        elif thinking and (thinking_emoji is not None):
            emoji, state = thinking_emoji, "THINKING"
        elif tongue and (tongue_emoji is not None):
            emoji, state = tongue_emoji, "TONGUE"
        elif smile and (evil_smile_emoji is not None):
            emoji, state = evil_smile_emoji, "SMILING"
        else:
            emoji, state = (staring_emoji if staring_emoji is not None else blank), "STARE"

        frame_r = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.putText(frame_r, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", frame_r)
        cv2.imshow("Emoji", emoji if emoji is not None else blank)

        # ===== Lyrics typewriter 1.9s per line =====
        if current_start_time is None:
            if not lyrics_queue:
                lyrics_queue = deque(lyrics_lines)  # loop
            if lyrics_queue:
                current_line = lyrics_queue.popleft()
            else:
                current_line = ""
            current_start_time = time.time()

        elapsed = time.time() - current_start_time
        progress = min(1.0, elapsed / LINE_TYPE_DURATION_SEC)
        chars_to_show = int(len(current_line) * progress)
        if progress >= 1.0:
            if current_line:
                lines_buffer.append(current_line)
                if len(lines_buffer) > 400:
                    lines_buffer.popleft()
            current_line = ""
            current_start_time = None

        render_lyrics(text_canvas, lines_buffer, current_line, chars_to_show)
        cv2.imshow("Lyrics", text_canvas)

        k = cv2.waitKey(15) & 0xFF
        if k in (27, ord('q')):
            break

# ========= Cleanup =========
cap.release()
cv2.destroyAllWindows()
try:
    if pygame_available:
        import pygame
        pygame.mixer.music.stop()
        pygame.mixer.quit()
except Exception:
    pass
