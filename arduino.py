import threading
import time
import queue

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

    Протокол (текстовый, \\n как разделитель):
      → GREEN     ← OK:GREEN   (свой номер — зелёный на HOLD_MS)
      → RED       ← OK:RED     (чужой номер — красный на HOLD_MS)
      → CLOSE     ← OK:CLOSE   (выключить оба)
      → PING      ← PONG

    Прошивка: barrier/barrier.ino. HOLD_MS = 3000 мс по умолчанию —
    Arduino сама гасит светодиод, Python не тратит ресурсы на таймеры.

    Для реального шлагбаума достаточно заменить светодиод на реле-модуль:
    протокол не меняется (GREEN = «замкнуть реле», CLOSE = «разомкнуть»).
    """

    def __init__(self):
        self.serial = None
        self.is_connected = False
        self._lock = threading.Lock()
        self._closing = False
        self._cmd_queue = queue.Queue()
        self._worker = None
        self._worker_stop = threading.Event()

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
                try:
                    self.serial.close()
                except Exception:
                    pass
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
        """Корректно завершает работу: останавливает worker, шлёт CLOSE, закрывает порт."""
        self._closing = True
        self._worker_stop.set()
        # Разбудить worker, если он ждёт в очереди
        self._cmd_queue.put((None, None))

        t = self._worker
        if t is not None and t.is_alive():
            try:
                t.join(timeout=5.0)
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
        """Отправляет команду и (опционально) ждёт ответа."""
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
        """Ставит команду в очередь — не блокирует UI."""
        if not ARDUINO_ENABLED or not PYSERIAL_AVAILABLE:
            return
        if self._closing:
            return
        self._ensure_worker()
        self._cmd_queue.put((cmd, expect))

    # ─────────────────────────────────────────────────────────────
    #  РАБОЧИЙ ПОТОК (один на всё время жизни контроллера)
    # ─────────────────────────────────────────────────────────────
    def _ensure_worker(self):
        if self._worker is None or not self._worker.is_alive():
            self._worker_stop.clear()
            self._worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="arduino-worker",
            )
            self._worker.start()

    def _worker_loop(self):
        while not self._worker_stop.is_set():
            try:
                item = self._cmd_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:  # sentinel — shutdown
                break

            cmd, expect = item
            if self._closing:
                continue

            with self._lock:
                if self.serial is None and not self.connect():
                    continue
                if self.serial is None:
                    continue
                ok = self._send_command(cmd, expect=expect, timeout=1.5)
                if not ok:
                    print(f"[WARN] Arduino не подтвердила {cmd}")

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