import threading
import time
import sys

try:
    import serial
    from serial.tools import list_ports
    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False

from config import (
    ARDUINO_PORT, ARDUINO_BAUDRATE, ARDUINO_ENABLED,
)


class ArduinoController:
    """Управление индикацией доступа (зелёный/красный светодиод).

    Протокол (текстовый, \n как разделитель):
      → GREEN     ← OK:GREEN   (свой номер — зелёный на 3 сек)
      → RED       ← OK:RED     (чужой номер — красный на 3 сек)
      → CLOSE     ← OK:CLOSE   (выключить оба)
      → PING      ← PONG

    Arduino сам гасит светодиод через HOLD_MS после включения.
    """

    def __init__(self):
        self.serial = None
        self.is_connected = False
        self._lock = threading.Lock()
        self._closing = False
        self._worker_thread = None

        if not ARDUINO_ENABLED:
            return
        if not PYSERIAL_AVAILABLE:
            return
        self.connect()

    # ─────────────────────────────────────────────────────────────
    #  ПОДКЛЮЧЕНИЕ
    # ─────────────────────────────────────────────────────────────
    def connect(self) -> bool:
        if not ARDUINO_ENABLED or not PYSERIAL_AVAILABLE:
            return False
        if self._closing:
            return False

        try:
            print(f"[INFO] Подключение к Arduino ({ARDUINO_PORT}, {ARDUINO_BAUDRATE} бод)...")
            self.serial = serial.Serial(
                port=ARDUINO_PORT,
                baudrate=ARDUINO_BAUDRATE,
                timeout=1.0,
                write_timeout=1.0,
            )
            # Arduino перезагружается при открытии порта — ждём
            time.sleep(2.0)
            self.serial.reset_input_buffer()

            # Проверяем связь командой PING
            if not self._send_command("PING", expect="PONG", timeout=2.0):
                print("[WARN] Arduino не ответила на PING")
                self.serial.close()
                self.serial = None
                return False

            self.is_connected = True
            print("[OK] Arduino подключена и отвечает.")
            return True

        except Exception as e:
            print(f"[ERROR] Не удалось подключиться к Arduino: {e}")
            self.is_connected = False
            if self.serial is not None:
                try:
                    self.serial.close()
                except Exception:
                    pass
                self.serial = None
            return False

    def disconnect(self):
        self._closing = True

        t = self._worker_thread
        if t is not None and t.is_alive():
            try:
                t.join(timeout=3.0)
            except Exception:
                pass

        with self._lock:
            if self.serial is not None:
                try:
                    if self.is_connected:
                        self._send_command("CLOSE", expect=None, timeout=0.5)
                except Exception:
                    pass
                try:
                    self.serial.close()
                except Exception:
                    pass
            self.serial = None
            self.is_connected = False

    # ─────────────────────────────────────────────────────────────
    #  ОТПРАВКА КОМАНД
    # ─────────────────────────────────────────────────────────────
    def _send_command(self, cmd: str, expect: str = None, timeout: float = 1.0) -> bool:
        """Отправляет команду и (опционально) ждёт ответа.
        Возвращает True, если команда отправлена и (если expect задан) ответ получен."""
        if self.serial is None:
            return False

        try:
            self.serial.reset_input_buffer()
            self.serial.write((cmd + "\n").encode("ascii"))
            self.serial.flush()

            if expect is None:
                return True

            deadline = time.time() + timeout
            while time.time() < deadline:
                line = self.serial.readline().decode("ascii", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith(expect):
                    return True
            return False

        except Exception as e:
            print(f"[WARN] Serial send '{cmd}': {e}")
            return False

    # ─────────────────────────────────────────────────────────────
    #  ИНДИКАЦИЯ ДОСТУПА
    # ─────────────────────────────────────────────────────────────
    def signal_green(self):
        """Свой номер — зажечь зелёный светодиод."""
        self._send_async("GREEN", expect="OK:GREEN")

    def signal_red(self):
        """Чужой номер — зажечь красный светодиод."""
        self._send_async("RED", expect="OK:RED")

    def _send_async(self, cmd: str, expect: str = None):
        """Отправляет команду в отдельном потоке, чтобы не блокировать UI."""
        if not ARDUINO_ENABLED or not PYSERIAL_AVAILABLE:
            return
        if self._closing:
            return
        if not self.is_connected or self.serial is None:
            if not self.connect():
                return

        def _worker():
            with self._lock:
                if self.serial is None or self._closing:
                    return
                ok = self._send_command(cmd, expect=expect, timeout=1.5)
                if not ok:
                    print(f"[WARN] Arduino не подтвердила {cmd}")

        thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread = thread
        thread.start()

    # ─────────────────────────────────────────────────────────────
    #  ДИАГНОСТИКА
    # ─────────────────────────────────────────────────────────────
    def get_status_text(self) -> str:
        if not ARDUINO_ENABLED:
            return "🔌 Arduino: отключена (симуляция)"
        if not PYSERIAL_AVAILABLE:
            return "🔌 Arduino: pyserial не установлен"
        if self.is_connected:
            return f"✅ Arduino: подключена ({ARDUINO_PORT})"
        return "❌ Arduino: не подключена"