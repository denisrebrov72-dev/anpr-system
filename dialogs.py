import cv2
import numpy as np
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from config import COLOR_CARD, COLOR_WARNING, COLOR_SUCCESS, COLOR_TEXT_DIM


class ManualChoiceDialog:
    """Диалог выбора варианта распознавания (для сомнительных случаев).

    Открывается, когда оба прохода OCR согласны (tie), но оба неуверенны.
    Оператор может: выбрать сырой, выбрать обработанный, ввести вручную,
    принять авто-выбор или отменить.

    После закрытия прочитать:
        dialog.result        — "raw" / "processed" / "manual" / None
        dialog.manual_plate  — если result == "manual", введённый текст
    """

    def __init__(self, parent, raw_cyr, proc_cyr, raw_crop, proc_crop,
                 raw_score, proc_score, auto_winner):
        self.result = None
        self.manual_plate = None
        self.auto_winner = auto_winner

        self.top = ctk.CTkToplevel(parent)
        self.top.title("Выбор варианта номера")
        self.top.geometry("980x680")
        self.top.transient(parent)
        self.top.grab_set()
        self.top.resizable(False, False)

        ctk.CTkLabel(
            self.top,
            text="⚠  Выберите правильный вариант или введите вручную:",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLOR_WARNING,
        ).pack(pady=18)

        cols = ctk.CTkFrame(self.top, fg_color="transparent")
        cols.pack(fill="both", expand=True, padx=20, pady=5)

        # ── Левая колонка: сырая вырезка ──
        left = ctk.CTkFrame(cols, fg_color=COLOR_CARD, corner_radius=12)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        ctk.CTkLabel(left, text="Сырая вырезка",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(16, 8))
        self._raw_photo = self._make_photo(raw_crop, (340, 120))
        ctk.CTkLabel(left, image=self._raw_photo, text="").pack(pady=10)
        ctk.CTkLabel(left, text=raw_cyr or "—",
                     font=ctk.CTkFont(size=24, weight="bold")).pack()
        ctk.CTkLabel(left, text=f"Балл: {raw_score:.2f}",
                     text_color=COLOR_TEXT_DIM,
                     font=ctk.CTkFont(size=12)).pack(pady=(0, 10))
        ctk.CTkButton(left, text="✅  Выбрать этот",
                      command=lambda: self._choose("raw"),
                      fg_color=COLOR_SUCCESS, hover_color="#15803d",
                      width=240, height=50,
                      font=ctk.CTkFont(size=14, weight="bold")).pack(pady=12)

        # ── Правая колонка: обработанная вырезка ──
        right = ctk.CTkFrame(cols, fg_color=COLOR_CARD, corner_radius=12)
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))
        ctk.CTkLabel(right, text="Обработанная (коррекция)",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(16, 8))
        self._proc_photo = self._make_photo(proc_crop, (340, 120))
        ctk.CTkLabel(right, image=self._proc_photo, text="").pack(pady=10)
        ctk.CTkLabel(right, text=proc_cyr or "—",
                     font=ctk.CTkFont(size=24, weight="bold")).pack()
        ctk.CTkLabel(right, text=f"Балл: {proc_score:.2f}",
                     text_color=COLOR_TEXT_DIM,
                     font=ctk.CTkFont(size=12)).pack(pady=(0, 10))
        ctk.CTkButton(right, text="✅  Выбрать этот",
                      command=lambda: self._choose("processed"),
                      fg_color=COLOR_SUCCESS, hover_color="#15803d",
                      width=240, height=50,
                      font=ctk.CTkFont(size=14, weight="bold")).pack(pady=12)

        # ── Ручной ввод ──
        manual_frame = ctk.CTkFrame(self.top, fg_color=COLOR_CARD, corner_radius=12)
        manual_frame.pack(fill="x", padx=20, pady=12)
        ctk.CTkLabel(manual_frame, text="✏️  Ввести номер вручную",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLOR_WARNING).pack(pady=(10, 6))
        entry_row = ctk.CTkFrame(manual_frame, fg_color="transparent")
        entry_row.pack(pady=(0, 12))
        self.manual_var = tk.StringVar()
        manual_entry = ctk.CTkEntry(
            entry_row, textvariable=self.manual_var,
            font=ctk.CTkFont(size=22, weight="bold"),
            width=340, height=50, justify="center",
            placeholder_text="А123ВС77")
        manual_entry.pack(side="left", padx=6)
        manual_entry.bind("<Return>", lambda e: self._accept_manual())
        ctk.CTkButton(entry_row, text="💾  Сохранить введённый",
                      command=self._accept_manual,
                      fg_color=COLOR_WARNING, hover_color="#d97706",
                      width=220, height=50,
                      font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=6)

        # Подпись авто-выбора
        if auto_winner == "raw":
            auto_label = "сырая"
        elif auto_winner in ("proc", "processed"):
            auto_label = "обработанная"
        else:  # tie
            auto_label = "не определён (оба прохода согласны) — по умолчанию сырая"

        ctk.CTkLabel(
            self.top,
            text=f"Автоматический выбор: {auto_label}",
            text_color=COLOR_TEXT_DIM,
            font=ctk.CTkFont(size=12, slant="italic"),
        ).pack(pady=(4, 0))

        btns = ctk.CTkFrame(self.top, fg_color="transparent")
        btns.pack(pady=14)
        ctk.CTkButton(btns, text="Принять авто-выбор",
                      command=self._accept_auto,
                      width=220, height=48,
                      font=ctk.CTkFont(size=13)).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="Отмена", command=self._cancel,
                      width=160, height=48,
                      fg_color="gray30", hover_color="gray20",
                      font=ctk.CTkFont(size=13)).pack(side="left", padx=6)

    def _make_photo(self, bgr_img, size):
        if bgr_img is None or bgr_img.size == 0:
            img = np.full((size[1], size[0], 3), 200, dtype=np.uint8)
        else:
            img = bgr_img
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail(size)
        return ImageTk.PhotoImage(pil)

    def _choose(self, which: str):
        """which: 'raw' или 'processed'."""
        self.result = which
        self.top.destroy()

    def _accept_auto(self):
        """Принять авто-выбор с корректным маппингом значений.

        'tie' в recognize_two_pass означает «оба прохода согласны» — при
        авто-выборе берём сырой как базовый (он не искажён обработкой).
        'proc' приводим к 'processed' — так ожидает app.py.
        """
        mapping = {
            "raw": "raw",
            "proc": "processed",
            "processed": "processed",
            "tie": "raw",
        }
        self.result = mapping.get(self.auto_winner, "raw")
        self.top.destroy()

    def _accept_manual(self):
        text = self.manual_var.get().strip()
        if not text:
            return
        self.result = "manual"
        self.manual_plate = text
        self.top.destroy()

    def _cancel(self):
        self.result = None
        self.top.destroy()