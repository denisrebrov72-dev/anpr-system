"""Абстракция звукового слоя. Работает на Windows, graceful fallback на других ОС."""

import os
import sys
import threading

_IS_WINDOWS = sys.platform.startswith("win")

if _IS_WINDOWS:
    try:
        import winsound
        _WINSOUND_AVAILABLE = True
    except ImportError:
        _WINSOUND_AVAILABLE = False
else:
    _WINSOUND_AVAILABLE = False


class SoundPlayer:
    """Плеер звуковых сигналов. На Windows использует winsound,
    на остальных ОС — заглушку (можно расширить позже под simpleaudio/playsound)."""

    def __init__(self, alarm_path=None):
        self.alarm_path = str(alarm_path) if alarm_path else None
        self._lock = threading.Lock()
        self._playing = False

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def is_available(self) -> bool:
        """Есть ли техническая возможность проигрывать звук."""
        if not _WINSOUND_AVAILABLE:
            return False
        if self.alarm_path and not os.path.isfile(self.alarm_path):
            return False
        return True

    def play_alarm(self) -> bool:
        """Проигрывает звук тревоги асинхронно. Возвращает True, если удалось запустить."""
        with self._lock:
            if self._playing:
                return False
            if not _WINSOUND_AVAILABLE:
                return False
            if not self.alarm_path or not os.path.isfile(self.alarm_path):
                return False
            try:
                winsound.PlaySound(
                    self.alarm_path,
                    winsound.SND_FILENAME | winsound.SND_ASYNC,
                )
                self._playing = True
                return True
            except Exception as e:
                print(f"[WARN] SoundPlayer.play_alarm: {e}")
                self._playing = False
                return False

    def stop(self):
        """Останавливает звук (если играет)."""
        with self._lock:
            self._playing = False
            if not _WINSOUND_AVAILABLE:
                return
            try:
                winsound.PlaySound(None, 0)
            except Exception:
                pass

    def mark_stopped(self):
        """Сбросить флаг «играет» без принудительной остановки (для таймера).
        Используется, когда звук был запущен с флагом 'играет 5 секунд'."""
        with self._lock:
            self._playing = False