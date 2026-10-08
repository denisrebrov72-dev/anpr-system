import customtkinter as ctk
import cv2
import torch
import os
import csv
import tkinter as tk
from pathlib import Path
from PIL import Image
from datetime import datetime
from tkinter import messagebox, filedialog
from config import (COLOR_CARD, COLOR_TEXT_DIM, COLOR_SUCCESS, COLOR_SUCCESS_HOVER,
    COLOR_DANGER, COLOR_STOP, COLOR_STOP_HOVER, COLOR_VIDEO_BTN, COLOR_VIDEO_BTN_HOVER,
    COLOR_CAMERA_BTN, COLOR_CAMERA_BTN_HOVER, COLOR_WATCH_BTN, COLOR_WATCH_BTN_HOVER,
    COLOR_REVIEW_BG, COLOR_REVIEW_BORDER, COLOR_REVIEW_TEXT, COLOR_REVIEW_DET_TEXT,
    COLOR_NEUTRAL, COLOR_NEUTRAL_HOVER, COLOR_SIDEBAR_HOVER, COLOR_TEXT_NORMAL,
    COLOR_LOG_REVIEWED_BG, COLOR_LOG_UNREVIEWED_BG, COLOR_REVIEWED_TAG,
    YOLO_MODEL_PATH, OCR_MODEL_PATH, CAMERA_INDEX,
    FRAMES_DIR, CROPS_DIR,
    ALLOWED_FRAMES_DIR, ALLOWED_CROPS_DIR,
    WATCH_FRAMES_DIR, WATCH_CROPS_DIR)
from inference import YOLODetector, CRNNRecognizer, ANPR_Pipeline
from utils import normalize_plate, to_cyrillic
from database import ( log_detection, is_allowed_plate, is_watch_plate, init_db,
    clear_log_with_photos, clear_allowed_with_photos,
    clear_watch_with_photos, clear_all_with_photos,
    delete_plate_with_photos, search_log,
    get_log_stats, get_allowed_list, get_watch_list,
    add_plate, add_watch_plate,
    deactivate_watch_plate,
    update_allowed_with_photos, update_watch_with_photos,
    get_log_list, update_plate_reviewed, get_unreviewed_count)
from dialogs import ManualChoiceDialog
from arduino import ArduinoController
from sound import SoundPlayer

ALARM_PATH = Path(__file__).parent / "fire-alarm.wav"
ALARM_DURATION_MS = 5000

_MAX_LOGGED_CACHE = 5000


class ANPRApp(ctk.CTkFrame):
    """Главный класс приложения ANPR."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.grid(row=0, column=0, sticky="nsew")
        self.parent.grid_columnconfigure(1, weight=1)
        self.parent.grid_rowconfigure(0, weight=1)

        self.is_running = False
        self.anpr_pipeline = None
        self.detector = None
        self.recognizer = None
        self.cap = None
        self.last_detected_plate = None
        self.last_detection_time = 0
        self._logged_track_ids = set()
        self._logged_plates = set()

        self.video_file_path = None
        self.is_video_mode = False

        self.review_active = False
        self.review_frame_path = None
        self.review_crop_path = None
        self.review_yolo_conf = 0.0
        self.review_detected_text = ""

        self._choice_dialog_open = False

        self.is_alarm_playing = False
        self.arduino = ArduinoController()
        self.sound = SoundPlayer(ALARM_PATH)

        self._log_unreviewed_only = False
        self.barrier_enabled_var = tk.BooleanVar(value=True)
        self._status_cache = {}

        init_db()
        self._create_sidebar()
        self._create_frames()
        self.show_frame("menu")

    # ════════════════════════════════════════════════════════════════
    #  БОКОВАЯ ПАНЕЛЬ
    # ════════════════════════════════════════════════════════════════
    def _create_sidebar(self):
        sidebar = ctk.CTkFrame(self.parent, width=200, corner_radius=0, fg_color=COLOR_CARD)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(sidebar, text="ANPR System",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=COLOR_TEXT_DIM).pack(pady=20)

        for text, frame_name in [("Главное меню", "menu"), ("Детекция авто", "detection"),
                                 ("Базы номеров", "access"), ("Проверка", "check")]:
            ctk.CTkButton(sidebar, text=text,
                          command=lambda fn=frame_name: self.show_frame(fn),
                          height=45, fg_color="transparent",
                          text_color=COLOR_TEXT_DIM,
                          hover_color=COLOR_SIDEBAR_HOVER).pack(fill="x", padx=10, pady=5)

    # ════════════════════════════════════════════════════════════════
    #  СОЗДАНИЕ ЭКРАНОВ
    # ════════════════════════════════════════════════════════════════
    def _create_frames(self):
        self.frames = {}
        self._stat_labels = {}
        self._create_menu_frame()
        self._create_detection_frame()
        self._create_access_frame()
        self._create_check_frame()

    # ── Экран 0: Главное меню ──────────────────────────────────────
    def _create_menu_frame(self):
        f = ctk.CTkFrame(self.parent, fg_color="transparent")
        f.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        ctk.CTkLabel(f, text="ГЛАВНОЕ МЕНЮ",
                     font=ctk.CTkFont(size=32, weight="bold")).pack(pady=(40, 20))

        stats_grid = ctk.CTkFrame(f, fg_color="transparent")
        stats_grid.pack(fill="x", padx=10, pady=10)
        stats_grid.grid_columnconfigure(0, weight=1)
        stats_grid.grid_columnconfigure(1, weight=1)

        def make_stat_card(parent, title, key):
            card = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=12)
            ctk.CTkLabel(card, text=title, text_color=COLOR_TEXT_DIM,
                         font=ctk.CTkFont(size=14)).pack(anchor="w")
            lbl = ctk.CTkLabel(card, text="—", font=ctk.CTkFont(size=28, weight="bold"))
            lbl.pack(anchor="w", pady=(4, 0))
            self._stat_labels[key] = lbl
            return card

        make_stat_card(stats_grid, "Всего детекций", "total").grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        make_stat_card(stats_grid, "Уникальных номеров", "unique").grid(row=0, column=1, padx=8, pady=8, sticky="ew")
        make_stat_card(stats_grid, "Сегодня", "today").grid(row=1, column=0, padx=8, pady=8, sticky="ew")
        make_stat_card(stats_grid, "Разрешённых авто", "allowed").grid(row=1, column=1, padx=8, pady=8, sticky="ew")
        make_stat_card(stats_grid, "В контроле", "watch").grid(row=2, column=0, padx=8, pady=8, sticky="ew")
        make_stat_card(stats_grid, "Непроверенных", "unreviewed").grid(row=2, column=1, padx=8, pady=8, sticky="ew")

        ctk.CTkButton(f, text="ПОЛНАЯ ОЧИСТКА ВСЕХ БАЗ", command=self._clear_all,
                      fg_color=COLOR_DANGER, hover_color=COLOR_STOP_HOVER,
                      height=60, width=320,
                      font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(30, 10))
        self.frames["menu"] = f

    # ── Экран 1: Детекция авто ─────────────────────────────────────
    def _create_detection_frame(self):
        f = ctk.CTkFrame(self.parent, fg_color="transparent")
        f.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        top = ctk.CTkFrame(f, fg_color="transparent")
        top.pack(fill="x", pady=(10, 10))

        ctk.CTkLabel(top, text="ДЕТЕКЦИЯ АВТО",
                     font=ctk.CTkFont(size=24, weight="bold")).pack(side="left")

        self.btn_det_start = ctk.CTkButton(top, text="▶  Старт",
                                           command=self._start_camera,
                                           fg_color=COLOR_SUCCESS, hover_color=COLOR_SUCCESS_HOVER,
                                           width=100, height=40)
        self.btn_det_start.pack(side="right", padx=(8, 0))

        self.btn_det_stop = ctk.CTkButton(top, text="⏹  Стоп", command=self._stop_camera,
                                          fg_color=COLOR_STOP, hover_color=COLOR_STOP_HOVER,
                                          width=100, height=40, state="disabled")
        self.btn_det_stop.pack(side="right", padx=(8, 0))

        self.btn_det_video = ctk.CTkButton(top, text="Выбрать видео", command=self._select_video,
                                           fg_color=COLOR_VIDEO_BTN, hover_color=COLOR_VIDEO_BTN_HOVER,
                                           width=160, height=40)
        self.btn_det_video.pack(side="right", padx=(8, 0))

        ctk.CTkButton(top, text="Камера", command=self._select_camera,
                      fg_color=COLOR_CAMERA_BTN, hover_color=COLOR_CAMERA_BTN_HOVER,
                      width=100, height=40).pack(side="right", padx=(8, 0))

        ctk.CTkButton(top, text="Очистить базу", command=self._clear_log,
                      fg_color=COLOR_DANGER, hover_color=COLOR_STOP_HOVER,
                      width=160, height=40).pack(side="right", padx=(8, 0))

        controls_row = ctk.CTkFrame(f, fg_color="transparent")
        controls_row.pack(fill="x", pady=(4, 6))

        self.lbl_video_source = ctk.CTkLabel(controls_row, text="Источник: камера",
                                              text_color=COLOR_TEXT_DIM,
                                              font=ctk.CTkFont(size=12))
        self.lbl_video_source.pack(side="left")

        self.chk_barrier = ctk.CTkCheckBox(
            controls_row,
            text="🚦  Показывать индикацию доступа (зелёный/красный)",
            variable=self.barrier_enabled_var,
            font=ctk.CTkFont(size=13, weight="bold"),
            checkbox_width=22, checkbox_height=22,
        )
        self.chk_barrier.pack(side="right")

        body = ctk.CTkFrame(f, fg_color="transparent")
        body.pack(fill="both", expand=True, pady=10)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        video_frame = ctk.CTkFrame(body, fg_color=COLOR_CARD, corner_radius=12)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.video_label = ctk.CTkLabel(video_frame, text="Камера выключена",
                                        text_color=COLOR_TEXT_DIM,
                                        font=ctk.CTkFont(size=16))
        self.video_label.pack(expand=True)

        result_frame = ctk.CTkScrollableFrame(body, fg_color=COLOR_CARD, corner_radius=12)
        result_frame.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(result_frame, text="ПОСЛЕДНИЙ РАСПОЗНАННЫЙ НОМЕР",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLOR_TEXT_DIM).pack(pady=(16, 8))

        self.lbl_det_plate = ctk.CTkLabel(result_frame, text="—",
                                          font=ctk.CTkFont(size=28, weight="bold"))
        self.lbl_det_plate.pack(pady=(0, 8))

        self.lbl_det_status = ctk.CTkLabel(result_frame, text="",
                                           font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_det_status.pack(pady=(0, 8))

        self.lbl_det_time = ctk.CTkLabel(result_frame, text="",
                                         text_color=COLOR_TEXT_DIM,
                                         font=ctk.CTkFont(size=12))
        self.lbl_det_time.pack()

        self.lbl_det_crop = ctk.CTkLabel(result_frame, text="")
        self.lbl_det_crop.pack(pady=10)

        self.review_frame = ctk.CTkFrame(result_frame, fg_color=COLOR_REVIEW_BG,
                                          corner_radius=8, border_width=2,
                                          border_color=COLOR_REVIEW_BORDER)
        ctk.CTkLabel(self.review_frame, text="⚠️  Проверьте номер!",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=COLOR_REVIEW_TEXT).pack(pady=(10, 5))

        self.lbl_review_detected = ctk.CTkLabel(self.review_frame, text="",
                                                font=ctk.CTkFont(size=14),
                                                text_color=COLOR_REVIEW_DET_TEXT)
        self.lbl_review_detected.pack(pady=(0, 8))

        ctk.CTkLabel(self.review_frame, text="Фото машины:",
                     font=ctk.CTkFont(size=12),
                     text_color=COLOR_TEXT_DIM).pack(anchor="w", padx=12)
        self.lbl_review_full = ctk.CTkLabel(self.review_frame, text="")
        self.lbl_review_full.pack(pady=(4, 8))

        ctk.CTkLabel(self.review_frame, text="Фото номера:",
                     font=ctk.CTkFont(size=12),
                     text_color=COLOR_TEXT_DIM).pack(anchor="w", padx=12)
        self.lbl_review_crop = ctk.CTkLabel(self.review_frame, text="")
        self.lbl_review_crop.pack(pady=(4, 8))

        ctk.CTkLabel(self.review_frame, text="Правильный номер:",
                     font=ctk.CTkFont(size=13),
                     text_color=COLOR_TEXT_DIM).pack(anchor="w", padx=12, pady=(8, 4))

        self.review_entry = ctk.CTkEntry(self.review_frame,
                                         placeholder_text="Например: А123АА777",
                                         width=260, height=40,
                                         font=ctk.CTkFont(size=16))
        self.review_entry.pack(padx=12, pady=(0, 10))

        review_btns = ctk.CTkFrame(self.review_frame, fg_color="transparent")
        review_btns.pack(pady=(0, 12))

        ctk.CTkButton(review_btns, text="✅  Сохранить", command=self._confirm_review,
                      fg_color=COLOR_SUCCESS, hover_color=COLOR_SUCCESS_HOVER,
                      width=120, height=36).pack(side="left", padx=(0, 8))

        ctk.CTkButton(review_btns, text="Пропустить", command=self._skip_review,
                      fg_color=COLOR_NEUTRAL, hover_color=COLOR_NEUTRAL_HOVER,
                      width=100, height=36).pack(side="left")

        self.frames["detection"] = f

    # ── Экран 2: Базы номеров ──────────────────────────────────────
    def _create_access_frame(self):
        f = ctk.CTkFrame(self.parent, fg_color="transparent")
        f.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        ctk.CTkLabel(f, text="БАЗЫ НОМЕРОВ",
                     font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(10, 15))

        cols = ctk.CTkFrame(f, fg_color="transparent")
        cols.pack(fill="both", expand=True)
        cols.grid_columnconfigure(0, weight=1, uniform="col")
        cols.grid_columnconfigure(1, weight=1, uniform="col")
        cols.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(cols, fg_color=COLOR_CARD, corner_radius=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        ctk.CTkLabel(left, text="✅  СВОИ НОМЕРА",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=COLOR_SUCCESS).pack(pady=(16, 8))

        ctrl_left = ctk.CTkFrame(left, fg_color="transparent")
        ctrl_left.pack(fill="x", padx=12, pady=(0, 8))

        self.entry_acc_plate = ctk.CTkEntry(ctrl_left,
                                             placeholder_text="Номер (например, А123АА777)",
                                             width=200, height=36)
        self.entry_acc_plate.pack(side="left", padx=(0, 8))

        ctk.CTkButton(ctrl_left, text="Добавить", command=self._add_allowed,
                      fg_color=COLOR_SUCCESS, hover_color=COLOR_SUCCESS_HOVER,
                      width=100, height=36).pack(side="left", padx=(0, 4))

        ctrl_left2 = ctk.CTkFrame(left, fg_color="transparent")
        ctrl_left2.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkButton(ctrl_left2, text="Удалить", command=self._remove_allowed,
                      fg_color=COLOR_STOP, hover_color=COLOR_STOP_HOVER,
                      width=100, height=32).pack(side="left", padx=(0, 4))

        ctk.CTkButton(ctrl_left2, text="Очистить", command=self._clear_allowed,
                      fg_color=COLOR_DANGER, hover_color=COLOR_STOP_HOVER,
                      width=100, height=32).pack(side="left")

        self._allowed_table = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self._allowed_table.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        right = ctk.CTkFrame(cols, fg_color=COLOR_CARD, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        ctk.CTkLabel(right, text="🚨  КОНТРОЛЬ",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=COLOR_WATCH_BTN).pack(pady=(16, 8))

        ctrl_right = ctk.CTkFrame(right, fg_color="transparent")
        ctrl_right.pack(fill="x", padx=12, pady=(0, 8))

        self.entry_watch = ctk.CTkEntry(ctrl_right,
                                         placeholder_text="Номер для контроля",
                                         width=200, height=36)
        self.entry_watch.pack(side="left", padx=(0, 8))

        ctk.CTkButton(ctrl_right, text="Поставить", command=self._add_to_watch,
                      fg_color=COLOR_WATCH_BTN, hover_color=COLOR_WATCH_BTN_HOVER,
                      width=110, height=36).pack(side="left", padx=(0, 4))

        ctrl_right2 = ctk.CTkFrame(right, fg_color="transparent")
        ctrl_right2.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkButton(ctrl_right2, text="Снять", command=self._remove_from_watch,
                      fg_color=COLOR_NEUTRAL, hover_color=COLOR_NEUTRAL_HOVER,
                      width=100, height=32).pack(side="left", padx=(0, 4))

        ctk.CTkButton(ctrl_right2, text="Очистить", command=self._clear_watch,
                      fg_color=COLOR_DANGER, hover_color=COLOR_STOP_HOVER,
                      width=100, height=32).pack(side="left")

        self._watch_table = ctk.CTkScrollableFrame(right, fg_color="transparent")
        self._watch_table.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.lbl_arduino = ctk.CTkLabel(f, text=self.arduino.get_status_text(),
                                        text_color=COLOR_TEXT_DIM,
                                        font=ctk.CTkFont(size=12))
        self.lbl_arduino.pack(pady=(8, 0))

        self.frames["access"] = f

    # ── Экран 3: Проверка ──────────────────────────────────────────
    def _create_check_frame(self):
        f = ctk.CTkFrame(self.parent, fg_color="transparent")
        f.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        ctk.CTkLabel(f, text="ПРОВЕРКА",
                     font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(10, 5))

        log_section = ctk.CTkFrame(f, fg_color=COLOR_CARD, corner_radius=12)
        log_section.pack(fill="x", pady=(5, 10))

        log_header = ctk.CTkFrame(log_section, fg_color="transparent")
        log_header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(log_header, text="ЖУРНАЛ ДЕТЕКЦИЙ",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLOR_TEXT_DIM).pack(side="left")

        log_btns = ctk.CTkFrame(log_header, fg_color="transparent")
        log_btns.pack(side="right")

        ctk.CTkButton(log_btns, text="Все записи", width=100, height=28,
                      command=self._show_all_logs).pack(side="left", padx=2)

        ctk.CTkButton(log_btns, text="⚠️ Непроверенные",
                      fg_color=COLOR_WATCH_BTN, hover_color=COLOR_WATCH_BTN_HOVER,
                      width=140, height=28,
                      command=self._show_unreviewed_logs).pack(side="left", padx=2)

        ctk.CTkButton(log_btns, text="Обновить", width=90, height=28,
                      command=self._refresh_log_list).pack(side="left", padx=2)

        ctk.CTkButton(log_btns, text="💾 Экспорт CSV", width=140, height=28,
                      fg_color=COLOR_SUCCESS, hover_color=COLOR_SUCCESS_HOVER,
                      command=self._export_log_csv).pack(side="left", padx=2)

        self._log_table = ctk.CTkScrollableFrame(log_section, fg_color="transparent", height=280)
        self._log_table.pack(fill="x", padx=10, pady=5)

        search_frame = ctk.CTkFrame(f, fg_color="transparent")
        search_frame.pack(fill="x", pady=5)

        self.entry_search = ctk.CTkEntry(search_frame,
                                         placeholder_text="Введите номер для поиска...",
                                         width=300, height=40,
                                         font=ctk.CTkFont(size=16))
        self.entry_search.pack(side="left", padx=(0, 10))

        ctk.CTkButton(search_frame, text="🔍  Найти",
                      command=self._search_plate,
                      width=120, height=40).pack(side="left")

        self._search_results = ctk.CTkScrollableFrame(f, fg_color=COLOR_CARD,
                                                      corner_radius=12,
                                                      label_text="Результаты поиска")
        self._search_results.pack(fill="both", expand=True, pady=10)

        self.frames["check"] = f

    # ════════════════════════════════════════════════════════════════
    #  ВЫБОР ИСТОЧНИКА ВИДЕО
    # ════════════════════════════════════════════════════════════════
    def _select_video(self):
        filename = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files", "*.*")])
        if not filename:
            return
        self.video_file_path = filename
        self.is_video_mode = True
        self.lbl_video_source.configure(text=f"Источник: {os.path.basename(filename)}")

    def _select_camera(self):
        self.video_file_path = None
        self.is_video_mode = False
        self.lbl_video_source.configure(text="Источник: камера")

    # ════════════════════════════════════════════════════════════════
    #  ЗВУКОВАЯ ТРЕВОГА
    # ════════════════════════════════════════════════════════════════
    def _play_alarm(self):
        if self.is_alarm_playing:
            return
        if not self.sound.play_alarm():
            return
        self.is_alarm_playing = True
        self.after(ALARM_DURATION_MS, self._on_alarm_timeout)

    def _on_alarm_timeout(self):
        self.is_alarm_playing = False
        self.sound.mark_stopped()

    # ════════════════════════════════════════════════════════════════
    #  СОХРАНЕНИЕ ФОТО
    # ════════════════════════════════════════════════════════════════
    def _save_photos(self, frame, crop_img, folder_frames, folder_crops):
        os.makedirs(str(folder_frames), exist_ok=True)
        os.makedirs(str(folder_crops), exist_ok=True)
        ts = datetime.now().isoformat()
        ts_safe = ts.replace(':', '-').replace('.', '-')
        frame_path = str(Path(folder_frames) / f"frame_{ts_safe}.jpg")
        crop_path = str(Path(folder_crops) / f"crop_{ts_safe}.jpg")
        cv2.imwrite(frame_path, frame)
        if crop_img is not None and crop_img.size > 0:
            cv2.imwrite(crop_path, crop_img)
        return frame_path, crop_path

    # ════════════════════════════════════════════════════════════════
    #  КАМЕРА / ВИДЕО
    # ════════════════════════════════════════════════════════════════
    def _start_camera(self):
        self.btn_det_start.configure(state="disabled")
        self.btn_det_stop.configure(state="normal")
        self.btn_det_video.configure(state="disabled")

        if self.anpr_pipeline is None:
            try:
                print("[INFO] Загрузка моделей ANPR...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.detector = YOLODetector(YOLO_MODEL_PATH, device)
                self.recognizer = CRNNRecognizer(OCR_MODEL_PATH, torch.device("cpu"))
                self.anpr_pipeline = ANPR_Pipeline(self.recognizer)
                print(f"[OK] Модели загружены на {device}!")
            except Exception as e:
                print(f"[ERROR] Не удалось загрузить модели: {e}")
                self._stop_camera()
                return

        if self.is_video_mode and self.video_file_path:
            self.cap = cv2.VideoCapture(self.video_file_path)
            print(f"[INFO] Открыт видеофайл: {self.video_file_path}")
        else:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            print(f"[INFO] Открыта камера (индекс {CAMERA_INDEX})")

        if not self.cap.isOpened():
            messagebox.showerror("Ошибка",
                "Не удалось открыть источник видео!\n"
                "Проверьте камеру или путь к файлу.")
            self._stop_camera()
            return

        self.is_running = True
        self.last_detected_plate = None
        self.last_detection_time = 0
        self._logged_track_ids = set()
        self._logged_plates = set()
        self._status_cache = {}
        self._frame_skip_counter = 0
        self._last_detections = []
        self._last_results = []
        self._camera_tick()

    def _stop_camera(self):
        self.is_running = False
        self.is_alarm_playing = False
        self.sound.stop()

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.btn_det_start.configure(state="normal")
        self.btn_det_stop.configure(state="disabled")
        self.btn_det_video.configure(state="normal")
        self.video_label.configure(text="Камера выключена", image=None)
        if self.review_active:
            self._hide_review()

    def _cv2_to_ctk(self, img, max_width=280):
        h, w = img.shape[:2]
        ratio = max_width / w
        new_w = max_width
        new_h = int(h * ratio)
        resized = cv2.resize(img, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return ctk.CTkImage(light_image=pil, dark_image=pil, size=(new_w, new_h))

    def _camera_tick(self):
        if not self.is_running:
            return
        if self.anpr_pipeline is None or self.detector is None:
            self._stop_camera()
            return
        ret, frame = self.cap.read()

        if not ret:
            if self.is_video_mode:
                print("[INFO] Видеофайл закончился.")
                self._stop_camera()
                self.is_video_mode = False
                self.video_file_path = None
                self.lbl_video_source.configure(text="Источник: камера")
            else:
                print("[WARN] Поток с камеры прерван.")
                self._stop_camera()
            return

        self._frame_skip_counter += 1

        if self._frame_skip_counter % 3 == 0 and not self._choice_dialog_open:
            try:
                self._last_detections = self.detector.track(frame)
            except Exception as e:
                print(f"[WARN] Ошибка детекции: {e}")
                self._last_detections = []
            try:
                self._last_results = self.anpr_pipeline.process_frame(frame, self._last_detections)
            except Exception as e:
                print(f"[WARN] Ошибка распознавания: {e}")
                self._last_results = []

        annotated_frame = frame.copy()
        if self._last_results:
            for r in self._last_results:
                plate_text = r.get("text", "")
                needs_review = r.get("needs_review", False)
                is_valid = r.get("is_valid", False)
                yolo_conf = r.get("confidence", 0.0)
                ocr_score = r.get("ocr_score", 0.0)
                box = r.get("bbox", None)
                if box and len(box) == 4:
                    x1, y1, x2, y2 = box
                    if needs_review:
                        rect_color = (0, 0, 255)
                        text_bg = (0, 0, 255)
                    else:
                        track_id_for_color = r.get("track_id")
                        cached = self._status_cache.get(track_id_for_color) if track_id_for_color is not None else None
                        if cached == "watch":
                            rect_color = (0, 215, 255)
                            text_bg = (0, 215, 255)
                        elif cached == "denied":
                            rect_color = (160, 160, 160)
                            text_bg = (160, 160, 160)
                        else:
                            rect_color = (0, 255, 0)
                            text_bg = (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), rect_color, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - 25), (x2, y1), text_bg, -1)
                    display_text = to_cyrillic(plate_text)
                    cv2.putText(annotated_frame, display_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    if self._frame_skip_counter % 3 == 0 and not self._choice_dialog_open:
                        crop_img = frame[y1:y2, x1:x2]
                        current_time = datetime.now().timestamp()
                        track_id = r.get("track_id")
                        self._handle_detection(plate_text, frame, crop_img,
                                               yolo_conf, needs_review, is_valid,
                                               current_time, track_id, ocr_score)
                        break

        h, w = annotated_frame.shape[:2]
        display_h = 480
        display_w = int(w * (display_h / h))
        resized = cv2.resize(annotated_frame, (display_w, display_h))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_img)
        img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil,
                              size=(display_w, display_h))
        self.video_label.configure(image=img_tk, text="")
        self.video_label.image = img_tk
        self.after(30, self._camera_tick)

    # ════════════════════════════════════════════════════════════════
    #  ОБРАБОТКА ДЕТЕКЦИИ
    # ════════════════════════════════════════════════════════════════
    def _cleanup_logged_cache(self):
        if len(self._logged_plates) > _MAX_LOGGED_CACHE:
            self._logged_plates.clear()
        if len(self._logged_track_ids) > _MAX_LOGGED_CACHE:
            self._logged_track_ids.clear()
        if len(self._status_cache) > _MAX_LOGGED_CACHE:
            self._status_cache.clear()

    def _handle_detection(self, plate_text, frame, crop_img, yolo_conf,
                          needs_review, is_valid, current_time,
                          track_id=None, ocr_score=0.0):
        plate_norm = normalize_plate(plate_text)

        if needs_review:
            if track_id is not None and track_id in self._logged_track_ids:
                return
            if plate_norm in self._logged_plates:
                return

            if not self._choice_dialog_open:
                try:
                    two_pass = self.anpr_pipeline.recognize_two_pass(crop_img)
                except Exception as e:
                    print(f"[WARN] Ошибка two-pass: {e}")
                    two_pass = None

                if two_pass is not None and two_pass.get("winner") == "tie" \
                        and two_pass.get("raw_score", 0.0) < 0.9 \
                        and two_pass.get("proc_score", 0.0) < 0.9:
                    self.last_detected_plate = plate_text
                    self.last_detection_time = current_time
                    if track_id is not None:
                        self._logged_track_ids.add(track_id)
                    self._logged_plates.add(plate_norm)
                    self._cleanup_logged_cache()
                    self._open_choice_dialog(two_pass, frame, crop_img, yolo_conf)
                    return

            self.last_detected_plate = plate_text
            self.last_detection_time = current_time
            self._show_review(plate_text, frame, crop_img, yolo_conf)
            if track_id is not None:
                self._logged_track_ids.add(track_id)
            self._logged_plates.add(plate_norm)
            self._cleanup_logged_cache()
            return

        if track_id is not None:
            if track_id in self._logged_track_ids:
                return
        else:
            if (plate_text == self.last_detected_plate and
                    current_time - self.last_detection_time < 5):
                return

        if plate_norm in self._logged_plates:
            return

        self.last_detected_plate = plate_text
        self.last_detection_time = current_time
        if track_id is not None:
            self._logged_track_ids.add(track_id)
        self._logged_plates.add(plate_norm)
        self._cleanup_logged_cache()
        if self.review_active:
            self._hide_review()

        barrier_opened = 0
        winner = "yolo+ocr"

        watch_info = is_watch_plate(plate_norm)
        if watch_info:
            if track_id is not None:
                self._status_cache[track_id] = "watch"
            self.lbl_det_plate.configure(text=to_cyrillic(plate_text),
                                         text_color=COLOR_REVIEW_TEXT)
            self.lbl_det_status.configure(text="🚨 КОНТРОЛЬ!",
                                          text_color=COLOR_REVIEW_TEXT)
            self.lbl_det_time.configure(text=datetime.now().strftime("%H:%M:%S"))
            self._play_alarm()
            winner = "watch"
            if not watch_info.get("photo_full") or not watch_info.get("photo_crop"):
                frame_path, crop_path = self._save_photos(
                    frame, crop_img, WATCH_FRAMES_DIR, WATCH_CROPS_DIR)
                update_watch_with_photos(plate_norm, frame_path, crop_path)
                self._refresh_watch_list()
        else:
            allowed_info = is_allowed_plate(plate_norm)
            if allowed_info:
                if track_id is not None:
                    self._status_cache[track_id] = "allowed"
                barrier_on = bool(self.barrier_enabled_var.get())
                if barrier_on:
                    self.arduino.signal_green()
                    barrier_opened = 1
                    self.lbl_det_status.configure(text="✅ РАЗРЕШЁН — зелёный",
                                                  text_color=COLOR_SUCCESS)
                    winner = "allowed+green"
                else:
                    self.lbl_det_status.configure(text="✅ РАЗРЕШЁН (индикация выкл)",
                                                  text_color=COLOR_SUCCESS)
                    winner = "allowed"
                self.lbl_det_plate.configure(text=to_cyrillic(plate_text),
                                             text_color=COLOR_TEXT_NORMAL)
                self.lbl_det_time.configure(text=datetime.now().strftime("%H:%M:%S"))

                if not allowed_info.get("photo_full") or not allowed_info.get("photo_crop"):
                    frame_path, crop_path = self._save_photos(
                        frame, crop_img, ALLOWED_FRAMES_DIR, ALLOWED_CROPS_DIR)
                    update_allowed_with_photos(plate_norm, frame_path, crop_path)
                    self._refresh_allowed_list()
            else:
                if track_id is not None:
                    self._status_cache[track_id] = "denied"
                self.arduino.signal_red()
                self.lbl_det_plate.configure(text=to_cyrillic(plate_text),
                                             text_color=COLOR_TEXT_NORMAL)
                self.lbl_det_status.configure(text="⛔ НЕТ ДОСТУПА — красный",
                                              text_color=COLOR_REVIEW_DET_TEXT)
                self.lbl_det_time.configure(text=datetime.now().strftime("%H:%M:%S"))
                winner = "denied"

        if crop_img is not None and crop_img.size > 0:
            thumb = self._cv2_to_ctk(crop_img, max_width=200)
            self.lbl_det_crop.configure(image=thumb, text="")
            self.lbl_det_crop.image = thumb
        else:
            self.lbl_det_crop.configure(image=None, text="Нет вырезки")

        self._log_detection(plate_text, frame, crop_img, yolo_conf,
                            winner, ocr_score, barrier_opened=barrier_opened)

    def _open_choice_dialog(self, two_pass, frame, crop_img, yolo_conf):
        self._choice_dialog_open = True
        dialog = ManualChoiceDialog(
            self,
            raw_cyr=two_pass.get("raw_cyr", ""),
            proc_cyr=two_pass.get("proc_cyr", ""),
            raw_crop=two_pass.get("raw_crop"),
            proc_crop=two_pass.get("proc_crop"),
            raw_score=two_pass.get("raw_score", 0.0),
            proc_score=two_pass.get("proc_score", 0.0),
            auto_winner=two_pass.get("winner", "raw"),
        )
        self._choice_context = {
            "dialog": dialog,
            "two_pass": two_pass,
            "frame": frame,
            "crop_img": crop_img,
            "yolo_conf": yolo_conf,
        }
        self._poll_choice_dialog()

    def _poll_choice_dialog(self):
        ctx = getattr(self, "_choice_context", None)
        if ctx is None:
            self._choice_dialog_open = False
            return
        dialog = ctx["dialog"]
        try:
            exists = dialog.top.winfo_exists()
        except Exception:
            exists = False
        if exists:
            self.after(150, self._poll_choice_dialog)
        else:
            self._choice_dialog_open = False
            self._on_choice_dialog_closed(ctx)

    def _on_choice_dialog_closed(self, ctx):
        dialog = ctx["dialog"]
        two_pass = ctx["two_pass"]
        frame = ctx["frame"]
        crop_img = ctx["crop_img"]
        yolo_conf = ctx["yolo_conf"]

        result = dialog.result

        if result is None:
            self.lbl_det_status.configure(text="⏭ Отменено",
                                          text_color=COLOR_TEXT_DIM)
            return

        if result == "manual":
            corrected = (dialog.manual_plate or "").strip()
            if not corrected:
                return
            plate_norm = normalize_plate(corrected)
            if not plate_norm:
                return
            add_plate(corrected, "allowed")
            frame_path, crop_path = self._save_photos(
                frame, crop_img, ALLOWED_FRAMES_DIR, ALLOWED_CROPS_DIR)
            update_allowed_with_photos(plate_norm, frame_path, crop_path)
            self._log_detection(corrected, frame, crop_img, yolo_conf,
                                "manual_correction", two_pass.get("raw_score", 0.0),
                                barrier_opened=0)
            self.lbl_det_plate.configure(text=to_cyrillic(corrected),
                                         text_color=COLOR_TEXT_NORMAL)
            self.lbl_det_status.configure(text="✅ Сохранён вручную",
                                          text_color=COLOR_SUCCESS)
            self.lbl_det_time.configure(text=datetime.now().strftime("%H:%M:%S"))
            if crop_img is not None and crop_img.size > 0:
                thumb = self._cv2_to_ctk(crop_img, max_width=200)
                self.lbl_det_crop.configure(image=thumb, text="")
                self.lbl_det_crop.image = thumb
            self._refresh_allowed_list()
            self._refresh_stats()
            return

        if result == "raw":
            chosen_text = two_pass.get("raw_text", "")
            chosen_score = two_pass.get("raw_score", 0.0)
            chosen_cyr = two_pass.get("raw_cyr", "")
            winner_tag = "two_pass_raw"
        elif result == "processed":
            chosen_text = two_pass.get("proc_text", "")
            chosen_score = two_pass.get("proc_score", 0.0)
            chosen_cyr = two_pass.get("proc_cyr", "")
            winner_tag = "two_pass_proc"
        else:
            return

        plate_norm = normalize_plate(chosen_text)
        barrier_opened = 0

        watch_info = is_watch_plate(plate_norm)
        if watch_info:
            self.lbl_det_status.configure(text="🚨 КОНТРОЛЬ!",
                                          text_color=COLOR_REVIEW_TEXT)
            self._play_alarm()
        else:
            allowed_info = is_allowed_plate(plate_norm)
            if allowed_info and self.barrier_enabled_var.get():
                self.arduino.signal_green()
                barrier_opened = 1
                self.lbl_det_status.configure(text="✅ РАЗРЕШЁН — зелёный",
                                              text_color=COLOR_SUCCESS)
            elif allowed_info:
                self.lbl_det_status.configure(text="✅ РАЗРЕШЁН (индикация выкл)",
                                              text_color=COLOR_SUCCESS)
            else:
                self.arduino.signal_red()
                self.lbl_det_status.configure(text="⛔ НЕТ ДОСТУПА — красный",
                                              text_color=COLOR_REVIEW_DET_TEXT)

        self.lbl_det_plate.configure(text=chosen_cyr or to_cyrillic(chosen_text),
                                     text_color=COLOR_TEXT_NORMAL)
        self.lbl_det_time.configure(text=datetime.now().strftime("%H:%M:%S"))

        if crop_img is not None and crop_img.size > 0:
            thumb = self._cv2_to_ctk(crop_img, max_width=200)
            self.lbl_det_crop.configure(image=thumb, text="")
            self.lbl_det_crop.image = thumb

        self._log_detection(chosen_text, frame, crop_img, yolo_conf,
                            winner_tag, chosen_score, barrier_opened=barrier_opened)
        self._refresh_stats()

    def _log_detection(self, plate_raw, frame, crop_img, yolo_conf, winner,
                       ocr_score, barrier_opened=0):
        ts = datetime.now().isoformat()
        plate_norm = normalize_plate(plate_raw)
        frame_path, crop_path = self._save_photos(frame, crop_img, FRAMES_DIR, CROPS_DIR)

        log_detection(plate_raw=plate_raw, plate_norm=plate_norm, ts=ts,
                      frame_path=frame_path, plate_crop_path=crop_path,
                      yolo_conf=yolo_conf, ocr_score=ocr_score, winner=winner,
                      mode="detection", barrier_opened=barrier_opened)

    # ════════════════════════════════════════════════════════════════
    #  СТАТИСТИКА
    # ════════════════════════════════════════════════════════════════
    def _refresh_stats(self):
        try:
            log_stats = get_log_stats()
            allowed_list = get_allowed_list()
            watch_list = get_watch_list()

            mapping = {
                "total": log_stats.get("total", 0),
                "unique": log_stats.get("unique", 0),
                "today": log_stats.get("today", 0),
                "allowed": len(allowed_list),
                "watch": len(watch_list),
                "unreviewed": log_stats.get("unreviewed", 0),
            }
            for key, value in mapping.items():
                lbl = self._stat_labels.get(key)
                if lbl:
                    lbl.configure(text=str(value))
        except Exception as e:
            print(f"[ERROR] Статистика: {e}")
            for lbl in self._stat_labels.values():
                lbl.configure(text="—")

    # ════════════════════════════════════════════════════════════════
    #  СПИСКИ
    # ════════════════════════════════════════════════════════════════
    def _refresh_allowed_list(self):
        for widget in self._allowed_table.winfo_children():
            widget.destroy()
        data = get_allowed_list()
        if not data:
            ctk.CTkLabel(self._allowed_table, text="Список пуст",
                         text_color=COLOR_TEXT_DIM,
                         font=ctk.CTkFont(size=13)).pack(pady=10)
            return
        for idx, row in enumerate(data, start=1):
            line = ctk.CTkFrame(self._allowed_table, fg_color="transparent")
            line.pack(fill="x", pady=2)
            ctk.CTkLabel(line, text=f"{idx}.", width=30,
                         text_color=COLOR_TEXT_DIM).pack(side="left")
            ctk.CTkLabel(line, text=row["plate_raw"],
                         font=ctk.CTkFont(weight="bold")).pack(side="left")

            crop_path = row.get("photo_crop")
            if crop_path and os.path.isfile(crop_path):
                try:
                    img = cv2.imread(crop_path)
                    if img is not None and img.size > 0:
                        thumb = self._cv2_to_ctk(img, max_width=60)
                        photo_lbl = ctk.CTkLabel(line, image=thumb, text="")
                        photo_lbl.image = thumb
                        photo_lbl.pack(side="left", padx=(8, 0))
                except Exception:
                    pass

            ctk.CTkLabel(line,
                         text=row.get("added_at", "")[:10] if row.get("added_at") else "",
                         text_color=COLOR_TEXT_DIM,
                         font=ctk.CTkFont(size=11)).pack(side="right")

    def _refresh_watch_list(self):
        for widget in self._watch_table.winfo_children():
            widget.destroy()
        data = get_watch_list()
        if not data:
            ctk.CTkLabel(self._watch_table, text="Список пуст",
                         text_color=COLOR_TEXT_DIM,
                         font=ctk.CTkFont(size=13)).pack(pady=10)
            return
        for idx, row in enumerate(data, start=1):
            line = ctk.CTkFrame(self._watch_table, fg_color="transparent")
            line.pack(fill="x", pady=2)
            ctk.CTkLabel(line, text=f"{idx}.", width=30,
                         text_color=COLOR_TEXT_DIM).pack(side="left")
            ctk.CTkLabel(line, text=row["plate_raw"],
                         font=ctk.CTkFont(weight="bold")).pack(side="left")

            crop_path = row.get("photo_crop")
            if crop_path and os.path.isfile(crop_path):
                try:
                    img = cv2.imread(crop_path)
                    if img is not None and img.size > 0:
                        thumb = self._cv2_to_ctk(img, max_width=60)
                        photo_lbl = ctk.CTkLabel(line, image=thumb, text="")
                        photo_lbl.image = thumb
                        photo_lbl.pack(side="left", padx=(8, 0))
                except Exception:
                    pass

            status = "🟢 активен" if row.get("active", 1) else "⚪ снят"
            ctk.CTkLabel(line, text=status, font=ctk.CTkFont(size=11),
                         text_color=COLOR_TEXT_DIM).pack(side="right")

    # ════════════════════════════════════════════════════════════════
    #  ОЧИСТКА БАЗ
    # ════════════════════════════════════════════════════════════════
    def _clear_log(self):
        if messagebox.askyesno("Подтверждение",
                                "Очистить Базу 1 (журнал детекций)?\nВсе фото будут удалены.",
                                icon="warning"):
            clear_log_with_photos()
            self._refresh_stats()
            if self.frames.get("check") and self.frames["check"].winfo_ismapped():
                self._refresh_log_list()
            messagebox.showinfo("Готово", "База 1 очищена.")

    def _clear_allowed(self):
        if messagebox.askyesno("Подтверждение",
                                "Очистить Базу 2 (наши номера)?",
                                icon="warning"):
            clear_allowed_with_photos()
            self._refresh_allowed_list()
            self._refresh_stats()
            messagebox.showinfo("Готово", "База 2 очищена.")

    def _clear_watch(self):
        if messagebox.askyesno("Подтверждение",
                                "Очистить Базу 3 (контроль)?",
                                icon="warning"):
            clear_watch_with_photos()
            self._refresh_watch_list()
            self._refresh_stats()
            messagebox.showinfo("Готово", "База 3 очищена.")

    def _clear_all(self):
        if messagebox.askyesno("ВНИМАНИЕ!",
                                "Удалить ВСЕ данные из ВСЕХ баз?\n\n"
                                "База 1 — журнал детекций + фото\n"
                                "База 2 — наши номера\n"
                                "База 3 — контроль\n\n"
                                "Это действие НЕЛЬЗЯ отменить!",
                                icon="warning"):
            clear_all_with_photos()
            self._refresh_stats()
            self._refresh_allowed_list()
            self._refresh_watch_list()
            if self.frames.get("check") and self.frames["check"].winfo_ismapped():
                self._refresh_log_list()
            messagebox.showinfo("Готово", "Все базы полностью очищены.")

    # ════════════════════════════════════════════════════════════════
    #  УПРАВЛЕНИЕ НОМЕРАМИ
    # ════════════════════════════════════════════════════════════════
    def _add_allowed(self):
        plate = self.entry_acc_plate.get().strip()
        if not plate:
            return
        add_plate(plate, "allowed")
        self.entry_acc_plate.delete(0, "end")
        self._refresh_allowed_list()
        self._refresh_stats()

    def _remove_allowed(self):
        plate = self.entry_acc_plate.get().strip()
        if not plate:
            return
        delete_plate_with_photos(plate)
        self.entry_acc_plate.delete(0, "end")
        self._refresh_allowed_list()
        self._refresh_stats()

    def _add_to_watch(self):
        plate = self.entry_watch.get().strip()
        if not plate:
            return
        plate_norm = normalize_plate(plate)
        add_watch_plate(plate, plate_norm, "manual")
        self.entry_watch.delete(0, "end")
        self._refresh_watch_list()
        self._refresh_stats()

    def _remove_from_watch(self):
        plate = self.entry_watch.get().strip()
        if not plate:
            return
        plate_norm = normalize_plate(plate)
        deactivate_watch_plate(plate_norm)
        self.entry_watch.delete(0, "end")
        self._refresh_watch_list()
        self._refresh_stats()

    # ════════════════════════════════════════════════════════════════
    #  ПОИСК
    # ════════════════════════════════════════════════════════════════
    def _search_plate(self):
        plate = self.entry_search.get().strip()
        if not plate:
            return

        results = search_log(plate)

        for widget in self._search_results.winfo_children():
            widget.destroy()

        if not results:
            ctk.CTkLabel(self._search_results, text="Ничего не найдено",
                         text_color=COLOR_TEXT_DIM,
                         font=ctk.CTkFont(size=16)).pack(pady=20)
            return

        seen = {}
        for row in results:
            pn = row.get("plate_norm", "")
            if pn not in seen:
                seen[pn] = row
            else:
                existing = seen[pn]
                existing_priority = {"watch": 3, "log": 2, "allowed": 1}
                if existing_priority.get(row.get("source", ""), 0) > \
                   existing_priority.get(existing.get("source", ""), 0):
                    seen[pn] = row

        for row in seen.values():
            source = row.get("source", "log")
            card = ctk.CTkFrame(self._search_results,
                                fg_color=COLOR_LOG_REVIEWED_BG, corner_radius=8)
            card.pack(fill="x", pady=4, padx=4)

            source_label = {
                "log": "📋 Журнал детекций",
                "allowed": "✅ Разрешённые",
                "watch": "🚨 Контроль",
            }.get(source, "Запись")

            plate_display = to_cyrillic(row.get('plate_raw', '—'))
            ctk.CTkLabel(card, text=f"{source_label}  |  {plate_display}",
                         font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=(8, 4))

            ts = row.get("timestamp", "") or row.get("added_at", "")
            if ts:
                ctk.CTkLabel(card, text=f"Время: {ts[:19].replace('T', ' ')}",
                             text_color=COLOR_TEXT_DIM,
                             font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10)

            if source == "log":
                conf = row.get("yolo_conf", 0)
                winner = row.get("winner", "—")
                reviewed = row.get("reviewed", 0)
                barrier = row.get("barrier_opened", 0)
                barrier_tag = "🚪 открыт" if barrier else "—"
                review_tag = "✅ проверен" if reviewed else "⚠️ непроверен"
                ctk.CTkLabel(card,
                             text=f"YOLO: {conf:.0%}  |  {winner}  |  🚪 {barrier_tag}  |  {review_tag}",
                             text_color=COLOR_TEXT_DIM,
                             font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10, pady=(0, 8))
            elif source == "watch":
                active = row.get("active", 1)
                ctk.CTkLabel(card,
                             text=f"Статус: {'🟢 активен' if active else '⚪ снят'}",
                             text_color=COLOR_TEXT_DIM,
                             font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10, pady=(0, 8))
            else:
                ctk.CTkLabel(card, text="",
                             font=ctk.CTkFont(size=1)).pack(anchor="w", padx=10, pady=(0, 8))

            photos_frame = ctk.CTkFrame(card, fg_color="transparent")
            photos_frame.pack(fill="x", padx=10, pady=(0, 8))

            if source == "log":
                crop_path = row.get("plate_crop_path", "")
                full_path = row.get("frame_path", "")
            else:
                crop_path = row.get("photo_crop", "")
                full_path = row.get("photo_full", "")

            if crop_path and os.path.isfile(crop_path):
                try:
                    img = cv2.imread(crop_path)
                    ctk_img = self._cv2_to_ctk(img, max_width=200)
                    lbl = ctk.CTkLabel(photos_frame, image=ctk_img, text="")
                    lbl.image = ctk_img
                    lbl.pack(side="left", padx=(0, 8))
                except Exception:
                    ctk.CTkLabel(photos_frame, text="[фото номера недоступно]",
                                 text_color=COLOR_TEXT_DIM).pack(side="left")
            else:
                ctk.CTkLabel(photos_frame, text="[нет фото номера]",
                             text_color=COLOR_TEXT_DIM).pack(side="left")

            if full_path and os.path.isfile(full_path):
                try:
                    img = cv2.imread(full_path)
                    ctk_img = self._cv2_to_ctk(img, max_width=200)
                    lbl = ctk.CTkLabel(photos_frame, image=ctk_img, text="")
                    lbl.image = ctk_img
                    lbl.pack(side="left")
                except Exception:
                    pass

    # ════════════════════════════════════════════════════════════════
    #  ЭКСПОРТ ЖУРНАЛА
    # ════════════════════════════════════════════════════════════════
    def _export_log_csv(self):
        try:
            rows = get_log_list(limit=10000, unreviewed_only=self._log_unreviewed_only)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось получить данные журнала:\n{e}")
            return

        if not rows:
            messagebox.showinfo("Экспорт", "Журнал пуст — нечего экспортировать.")
            return

        default_name = f"anpr_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = filedialog.asksaveasfilename(
            title="Сохранить журнал как CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not file_path:
            return

        columns = [
            "id", "timestamp", "plate_raw", "plate_norm",
            "yolo_conf", "ocr_score", "winner", "barrier_opened",
            "reviewed", "mode", "frame_path", "plate_crop_path",
        ]

        try:
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(columns)
                for row in rows:
                    writer.writerow([row.get(col, "") for col in columns])
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить CSV:\n{e}")
            return

        messagebox.showinfo("Готово", f"Журнал сохранён:\n{file_path}")

    # ════════════════════════════════════════════════════════════════
    #  ПАНЕЛЬ ПРОВЕРКИ
    # ════════════════════════════════════════════════════════════════
    def _show_review(self, detected_text, full_frame, crop_img, yolo_conf):
        self.review_active = True
        self.review_detected_text = detected_text
        self.review_yolo_conf = yolo_conf

        ts = datetime.now().isoformat()
        ts_safe = ts.replace(':', '-').replace('.', '-')

        self.review_frame_path = str(Path(FRAMES_DIR) / f"review_{ts_safe}.jpg")
        self.review_crop_path = str(Path(CROPS_DIR) / f"review_{ts_safe}.jpg")

        cv2.imwrite(self.review_frame_path, full_frame)
        if crop_img is not None and crop_img.size > 0:
            cv2.imwrite(self.review_crop_path, crop_img)

        self.lbl_review_detected.configure(
            text=f"Распознано: {to_cyrillic(detected_text)}  (уверенность: {yolo_conf:.0%})")

        if full_frame is not None and full_frame.size > 0:
            full_ctk = self._cv2_to_ctk(full_frame, max_width=260)
            self.lbl_review_full.configure(image=full_ctk, text="")
            self.lbl_review_full.image = full_ctk

        if crop_img is not None and crop_img.size > 0:
            crop_ctk = self._cv2_to_ctk(crop_img, max_width=260)
            self.lbl_review_crop.configure(image=crop_ctk, text="")
            self.lbl_review_crop.image = crop_ctk
        else:
            self.lbl_review_crop.configure(image=None, text="Нет фото номера")

        self.review_entry.delete(0, "end")
        self.review_entry.insert(0, to_cyrillic(detected_text))

        self.lbl_det_plate.configure(text=to_cyrillic(detected_text),
                                     text_color=COLOR_REVIEW_TEXT)
        self.lbl_det_status.configure(text="⚠️ Проверьте номер!",
                                      text_color=COLOR_REVIEW_TEXT)
        self.lbl_det_time.configure(text=datetime.now().strftime("%H:%M:%S"))

        self.review_frame.pack(pady=10, fill="x", padx=8)

    def _confirm_review(self):
        corrected = self.review_entry.get().strip()
        if not corrected:
            messagebox.showwarning("Внимание", "Введите правильный номер")
            return

        plate_norm = normalize_plate(corrected)
        if not plate_norm:
            messagebox.showwarning("Внимание",
                                    "Номер содержит только недопустимые символы")
            return

        add_plate(corrected, "allowed")

        frame_path = self.review_frame_path or ""
        crop_path = self.review_crop_path or ""
        if frame_path and crop_path:
            update_allowed_with_photos(plate_norm, frame_path, crop_path)

        log_detection(
            plate_raw=corrected, plate_norm=plate_norm,
            ts=datetime.now().isoformat(),
            frame_path=self.review_frame_path or "",
            plate_crop_path=self.review_crop_path or "",
            yolo_conf=self.review_yolo_conf, ocr_score=0.0,
            winner="manual_correction", mode="detection",
            barrier_opened=0
        )

        self.lbl_det_plate.configure(text=to_cyrillic(corrected),
                                     text_color=COLOR_TEXT_NORMAL)
        self.lbl_det_status.configure(text="✅ Сохранён в базу",
                                      text_color=COLOR_SUCCESS)
        self._hide_review()
        self._refresh_allowed_list()
        self._refresh_stats()

    def _skip_review(self):
        if self.review_detected_text:
            plate_norm = normalize_plate(self.review_detected_text)
            log_detection(
                plate_raw=self.review_detected_text, plate_norm=plate_norm,
                ts=datetime.now().isoformat(),
                frame_path=self.review_frame_path or "",
                plate_crop_path=self.review_crop_path or "",
                yolo_conf=self.review_yolo_conf, ocr_score=0.0,
                winner="skipped", mode="detection",
                barrier_opened=0
            )
        self.lbl_det_status.configure(text="⏭ Пропущен", text_color=COLOR_TEXT_DIM)
        self._hide_review()
        self._refresh_stats()

    def _hide_review(self):
        self.review_active = False
        self.review_frame.pack_forget()

    # ════════════════════════════════════════════════════════════════
    #  ЖУРНАЛ
    # ════════════════════════════════════════════════════════════════
    def _show_all_logs(self):
        self._log_unreviewed_only = False
        self._refresh_log_list()

    def _show_unreviewed_logs(self):
        self._log_unreviewed_only = True
        self._refresh_log_list()

    def _refresh_log_list(self):
        for widget in self._log_table.winfo_children():
            widget.destroy()

        data = get_log_list(limit=50, unreviewed_only=self._log_unreviewed_only)

        if not data:
            text = "Нет непроверенных записей" if self._log_unreviewed_only else "Журнал пуст"
            ctk.CTkLabel(self._log_table, text=text,
                         text_color=COLOR_TEXT_DIM,
                         font=ctk.CTkFont(size=14)).pack(pady=20)
            return

        for row in data:
            reviewed = row.get("reviewed", 0)
            bg_color = COLOR_LOG_REVIEWED_BG if reviewed else COLOR_LOG_UNREVIEWED_BG
            border_color = None if reviewed else COLOR_STOP

            card = ctk.CTkFrame(self._log_table, fg_color=bg_color, corner_radius=8,
                                border_width=1 if not reviewed else 0,
                                border_color=border_color)
            card.pack(fill="x", pady=3, padx=4)

            top_row = ctk.CTkFrame(card, fg_color="transparent")
            top_row.pack(fill="x", padx=10, pady=(8, 4))

            plate_text = row.get("plate_raw", "—")
            ctk.CTkLabel(top_row, text=to_cyrillic(plate_text),
                         font=ctk.CTkFont(size=18, weight="bold"),
                         text_color=COLOR_TEXT_NORMAL if reviewed else COLOR_REVIEW_TEXT
                         ).pack(side="left")

            ts = row.get("timestamp", "")
            ts_short = ts[:19].replace("T", " ") if ts else ""
            ctk.CTkLabel(top_row, text=ts_short, text_color=COLOR_TEXT_DIM,
                         font=ctk.CTkFont(size=11)).pack(side="right")

            mid_row = ctk.CTkFrame(card, fg_color="transparent")
            mid_row.pack(fill="x", padx=10, pady=(0, 4))

            conf = row.get("yolo_conf", 0)
            winner = row.get("winner", "—")
            barrier = row.get("barrier_opened", 0)
            barrier_label = "🚪 открыт" if barrier else "—"

            ctk.CTkLabel(mid_row,
                         text=f"YOLO: {conf:.0%}  |  {winner}  |  {barrier_label}",
                         text_color=COLOR_TEXT_DIM,
                         font=ctk.CTkFont(size=11)).pack(side="left")

            review_tag = "✅ проверен" if reviewed else "⚠️ НЕ ПРОВЕРЕН"
            ctk.CTkLabel(mid_row, text=review_tag,
                         text_color=COLOR_REVIEWED_TAG if reviewed else COLOR_REVIEW_DET_TEXT,
                         font=ctk.CTkFont(size=11, weight="bold")).pack(side="right")

            photos_frame = ctk.CTkFrame(card, fg_color="transparent")
            photos_frame.pack(fill="x", padx=10, pady=(4, 8))

            crop_path = row.get("plate_crop_path", "")
            if crop_path and os.path.isfile(crop_path):
                try:
                    img = cv2.imread(crop_path)
                    ctk_img = self._cv2_to_ctk(img, max_width=120)
                    lbl = ctk.CTkLabel(photos_frame, image=ctk_img, text="")
                    lbl.image = ctk_img
                    lbl.pack(side="left", padx=(0, 8))
                except Exception:
                    ctk.CTkLabel(photos_frame, text="[нет фото]",
                                 text_color=COLOR_TEXT_DIM,
                                 font=ctk.CTkFont(size=10)).pack(side="left")

            if not reviewed:
                action_row = ctk.CTkFrame(card, fg_color="transparent")
                action_row.pack(fill="x", padx=10, pady=(0, 8))

                entry_var = tk.StringVar(value=to_cyrillic(plate_text))
                entry = ctk.CTkEntry(action_row, textvariable=entry_var,
                                     width=200, height=32,
                                     font=ctk.CTkFont(size=14))
                entry.pack(side="left", padx=(0, 8))

                log_id = row.get("id")

                def _confirm(log_id=log_id, entry_var=entry_var, row=row):
                    corrected = entry_var.get().strip()
                    if not corrected:
                        return
                    plate_norm = normalize_plate(corrected)
                    if plate_norm:
                        add_plate(corrected, "allowed")
                        crop_p = row.get("plate_crop_path", "")
                        frame_p = row.get("frame_path", "")
                        if crop_p and frame_p:
                            update_allowed_with_photos(plate_norm, frame_p, crop_p)
                    update_plate_reviewed(log_id, corrected, plate_norm)
                    self._refresh_log_list()
                    self._refresh_allowed_list()
                    self._refresh_stats()

                def _skip(log_id=log_id):
                    update_plate_reviewed(log_id)
                    self._refresh_log_list()
                    self._refresh_stats()

                ctk.CTkButton(action_row, text="✅ Подтвердить",
                              command=_confirm,
                              fg_color=COLOR_SUCCESS, hover_color=COLOR_SUCCESS_HOVER,
                              width=120, height=32,
                              font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(0, 4))
                ctk.CTkButton(action_row, text="Пропустить", command=_skip,
                              fg_color=COLOR_NEUTRAL, hover_color=COLOR_NEUTRAL_HOVER,
                              width=100, height=32,
                              font=ctk.CTkFont(size=12)).pack(side="left")

    # ════════════════════════════════════════════════════════════════
    #  НАВИГАЦИЯ
    # ════════════════════════════════════════════════════════════════
    def show_frame(self, frame_name):
        if self.is_running:
            self._stop_camera()
        for frame in self.frames.values():
            frame.grid_remove()
        if frame_name in self.frames:
            self.frames[frame_name].grid()
        if frame_name == "menu":
            self._refresh_stats()
        if frame_name == "access":
            self._refresh_allowed_list()
            self._refresh_watch_list()
            self._refresh_stats()
            self.lbl_arduino.configure(text=self.arduino.get_status_text())
        if frame_name == "check":
            self._refresh_log_list()
            self._refresh_stats()

    # ════════════════════════════════════════════════════════════════
    #  КОРРЕКТНОЕ ЗАВЕРШЕНИЕ
    # ════════════════════════════════════════════════════════════════
    def on_close(self):
        try:
            if self.is_running:
                self._stop_camera()
        except Exception as e:
            print(f"[ERROR] on_close stop_camera: {e}")

        try:
            self.is_alarm_playing = False
            self.sound.stop()
        except Exception:
            pass

        try:
            self.arduino.disconnect()
        except Exception as e:
            print(f"[ERROR] on_close arduino.disconnect: {e}")