import os
from pathlib import Path

# ════════════════════════════════════════════════════════════════
#  БАЗОВЫЕ ПУТИ
BASE_DIR = Path(__file__).resolve().parent
CAMERA_INDEX = 0   # 0 = встроенная веб-камера, 1 = внешняя (обычно)
YOLO_MODEL_PATH = str(BASE_DIR / "models" / "yolo" / "model" / "best.pt")
OCR_MODEL_PATH = str(BASE_DIR / "models" / "ocr_crnn" / "quant" / "crnn_ocr_model_int8_fx.pth")

# ════════════════════════════════════════════════════════════════
#  ПАПКИ И ФАЙЛЫ
DEBUG_DIR = BASE_DIR / "debug"
LOG_DIR = BASE_DIR / "logs"
OURS_DIR = BASE_DIR / "ours"
WATCH_DIR = BASE_DIR / "watch"
EXPORTS_DIR = BASE_DIR / "exports"
DB_FILE = BASE_DIR / "anpr.db"

FRAMES_DIR = BASE_DIR / "frames"
CROPS_DIR = BASE_DIR / "crops"
ALLOWED_FRAMES_DIR = BASE_DIR / "allowed_frames"
ALLOWED_CROPS_DIR = BASE_DIR / "allowed_crops"
WATCH_FRAMES_DIR = BASE_DIR / "watch_frames"
WATCH_CROPS_DIR = BASE_DIR / "watch_crops"

for d in [DEBUG_DIR, LOG_DIR, OURS_DIR, WATCH_DIR, EXPORTS_DIR,
            FRAMES_DIR, CROPS_DIR,
            ALLOWED_FRAMES_DIR, ALLOWED_CROPS_DIR,
            WATCH_FRAMES_DIR, WATCH_CROPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════
#  ПАРАМЕТРЫ МОДЕЛИ
OCR_IMG_HEIGHT = 32
OCR_IMG_WIDTH = 128
OCR_ALPHABET = '0123456789ABCEHKMOPTXY'
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# ════════════════════════════════════════════════════════════════
#  ПАРАМЕТРЫ ОБРАБОТКИ ВИДЕО
FRAME_SKIP = 3          # обрабатывать каждый N-й кадр (для скорости)
PLATE_PADDING_RATIO = 0.15

# ════════════════════════════════════════════════════════════════
#  ЦВЕТА ДЛЯ ИНТЕРФЕЙСА
COLOR_PANEL = "#1f2937"
COLOR_TEXT = "#9ca3af"
COLOR_TEXT_DIM = "#8B8E99"
COLOR_CARD = "#2A2D3A"
COLOR_SUCCESS = "#22C55E"
COLOR_ACCENT = "#3b82f6"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#EF4444"

COLOR_STOP = "#AA3333"
COLOR_STOP_HOVER = "#CC4444"
COLOR_SUCCESS_HOVER = "#15803d"

COLOR_VIDEO_BTN = "#5555AA"
COLOR_VIDEO_BTN_HOVER = "#6666CC"

COLOR_CAMERA_BTN = "#558855"
COLOR_CAMERA_BTN_HOVER = "#66AA66"

COLOR_WATCH_BTN = "#AA6633"
COLOR_WATCH_BTN_HOVER = "#CC7744"

COLOR_REVIEW_BG = "#2A1A1A"
COLOR_REVIEW_BORDER = "#AA3333"
COLOR_REVIEW_TEXT = "#FF4444"
COLOR_REVIEW_DET_TEXT = "#FF6666"

COLOR_NEUTRAL = "gray30"
COLOR_NEUTRAL_HOVER = "gray20"
COLOR_SIDEBAR_HOVER = "#3a3a3a"
COLOR_TEXT_NORMAL = "#FFFFFF"

COLOR_LOG_REVIEWED_BG = "#2B2B2B"
COLOR_LOG_UNREVIEWED_BG = "#3A2020"
COLOR_REVIEWED_TAG = "#10B981"

# ════════════════════════════════════════════════════════════════
#  РАСШИРЕНИЯ ФАЙЛОВ
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

# ════════════════════════════════════════════════════════════════
#  НАСТРОЙКИ ARDUINO (индикация доступа: зелёный/красный светодиод)
ARDUINO_PORT = "COM3"
ARDUINO_BAUDRATE = 9600
ARDUINO_ENABLED = True