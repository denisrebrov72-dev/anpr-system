import sqlite3
import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from config import DB_FILE
from utils import normalize_plate

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# СХЕМА БАЗЫ ДАННЫХ
# -----------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS plates_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_raw TEXT NOT NULL,
    plate_norm TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    frame_path TEXT NOT NULL,
    plate_crop_path TEXT NOT NULL,
    yolo_conf REAL NOT NULL,
    ocr_score REAL NOT NULL,
    winner TEXT NOT NULL,
    reviewed INTEGER DEFAULT 0,
    mode TEXT DEFAULT 'detection',
    barrier_opened INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS plates_allowed (
    plate_norm TEXT PRIMARY KEY,
    plate_raw TEXT NOT NULL,
    added_at TEXT NOT NULL,
    notes TEXT,
    photo_full TEXT,
    photo_crop TEXT
);

CREATE TABLE IF NOT EXISTS plates_watch (
    plate_norm TEXT PRIMARY KEY,
    plate_raw TEXT NOT NULL,
    added_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    active INTEGER DEFAULT 1,
    photo_full TEXT,
    photo_crop TEXT
);

CREATE INDEX IF NOT EXISTS idx_log_timestamp ON plates_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_log_plate_norm ON plates_log(plate_norm);
CREATE INDEX IF NOT EXISTS idx_watch_active ON plates_watch(active);
"""


# -----------------------------------------------------------------------------
# ПОДКЛЮЧЕНИЕ
# -----------------------------------------------------------------------------
@contextmanager
def open_conn():
    conn = sqlite3.connect(str(DB_FILE), timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.DatabaseError as e:
            logger.warning("Не удалось установить PRAGMA: %s", e)
        yield conn
    finally:
        conn.close()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_FILE), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
    except sqlite3.DatabaseError:
        pass
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.DatabaseError:
        return False
    return any(r["name"] == column for r in rows)


def init_db():
    with open_conn() as conn:
        conn.executescript(SCHEMA_SQL)

        migrations = [
            ("plates_allowed", "photo_full", "TEXT"),
            ("plates_allowed", "photo_crop", "TEXT"),
            ("plates_watch", "photo_full", "TEXT"),
            ("plates_watch", "photo_crop", "TEXT"),
            ("plates_log", "reviewed", "INTEGER DEFAULT 0"),
            ("plates_log", "mode", "TEXT DEFAULT 'detection'"),
            ("plates_log", "barrier_opened", "INTEGER DEFAULT 0"),
        ]
        for table, column, coltype in migrations:
            if not _column_exists(conn, table, column):
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
                except sqlite3.DatabaseError as e:
                    logger.warning("ALTER TABLE %s ADD COLUMN %s: %s", table, column, e)
        conn.commit()


# -----------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ — УДАЛЕНИЕ ФОТО
# -----------------------------------------------------------------------------
def _delete_photo_files(paths: list):
    for p in paths:
        if p and os.path.isfile(p):
            try:
                os.remove(p)
            except OSError as e:
                logger.warning("Не удалось удалить файл %s: %s", p, e)


def _delete_folder_contents(folder):
    if not folder:
        return
    folder = str(folder)
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            full = os.path.join(folder, f)
            if os.path.isfile(full):
                try:
                    os.remove(full)
                except OSError as e:
                    logger.warning("Не удалось удалить файл %s: %s", full, e)


def _collect_photos_from_rows(rows) -> list:
    paths = []
    for r in rows:
        try:
            if "frame_path" in r.keys():
                paths.append(r["frame_path"])
            if "plate_crop_path" in r.keys():
                paths.append(r["plate_crop_path"])
            if "photo_full" in r.keys():
                paths.append(r["photo_full"])
            if "photo_crop" in r.keys():
                paths.append(r["photo_crop"])
        except Exception:
            continue
    return paths


# -----------------------------------------------------------------------------
# ОЧИСТКА БАЗ
# -----------------------------------------------------------------------------
def clear_log_with_photos():
    with open_conn() as conn:
        rows = conn.execute(
            "SELECT frame_path, plate_crop_path FROM plates_log"
        ).fetchall()
        paths = _collect_photos_from_rows(rows)
        conn.execute("DELETE FROM plates_log")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='plates_log'")
        conn.commit()
    _delete_photo_files(paths)


def clear_allowed_with_photos():
    with open_conn() as conn:
        rows = conn.execute(
            "SELECT photo_full, photo_crop FROM plates_allowed"
        ).fetchall()
        paths = _collect_photos_from_rows(rows)
        conn.execute("DELETE FROM plates_allowed")
        conn.commit()
    _delete_photo_files(paths)


def clear_watch_with_photos():
    with open_conn() as conn:
        rows = conn.execute(
            "SELECT photo_full, photo_crop FROM plates_watch"
        ).fetchall()
        paths = _collect_photos_from_rows(rows)
        conn.execute("DELETE FROM plates_watch")
        conn.commit()
    _delete_photo_files(paths)


def clear_all_with_photos():
    with open_conn() as conn:
        log_rows = conn.execute(
            "SELECT frame_path, plate_crop_path FROM plates_log"
        ).fetchall()
        allowed_rows = conn.execute(
            "SELECT photo_full, photo_crop FROM plates_allowed"
        ).fetchall()
        watch_rows = conn.execute(
            "SELECT photo_full, photo_crop FROM plates_watch"
        ).fetchall()

        paths = []
        paths.extend(_collect_photos_from_rows(log_rows))
        paths.extend(_collect_photos_from_rows(allowed_rows))
        paths.extend(_collect_photos_from_rows(watch_rows))

        conn.execute("DELETE FROM plates_log")
        conn.execute("DELETE FROM plates_allowed")
        conn.execute("DELETE FROM plates_watch")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='plates_log'")
        conn.commit()

    _delete_photo_files(paths)


# -----------------------------------------------------------------------------
# ФОТО — ОБНОВЛЕНИЕ ЗАПИСЕЙ
# -----------------------------------------------------------------------------
def update_allowed_with_photos(plate_norm: str, photo_full: str, photo_crop: str):
    plate_norm = normalize_plate(plate_norm)
    with open_conn() as conn:
        conn.execute(
            "UPDATE plates_allowed SET photo_full=?, photo_crop=? WHERE plate_norm=?",
            (photo_full, photo_crop, plate_norm),
        )
        conn.commit()


def update_watch_with_photos(plate_norm: str, photo_full: str, photo_crop: str):
    plate_norm = normalize_plate(plate_norm)
    with open_conn() as conn:
        conn.execute(
            "UPDATE plates_watch SET photo_full=?, photo_crop=? WHERE plate_norm=?",
            (photo_full, photo_crop, plate_norm),
        )
        conn.commit()


# -----------------------------------------------------------------------------
# БАЗА 1 — ЖУРНАЛ ДЕТЕКЦИЙ
# -----------------------------------------------------------------------------
def log_detection(plate_raw, plate_norm, ts, frame_path, plate_crop_path,
                  yolo_conf, ocr_score, winner, mode="detection",
                  barrier_opened: int = 0) -> bool:
    try:
        with open_conn() as conn:
            conn.execute(
                """INSERT INTO plates_log
                (plate_raw, plate_norm, timestamp, frame_path, plate_crop_path,
                yolo_conf, ocr_score, winner, mode, barrier_opened)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (plate_raw, plate_norm, ts, frame_path, plate_crop_path,
                 yolo_conf, ocr_score, winner, mode, int(barrier_opened)),
            )
            conn.commit()
        return True
    except sqlite3.DatabaseError as e:
        logger.exception("log_detection failed: %s", e)
        return False


def search_log(plate: str) -> List[Dict[str, Any]]:
    plate_norm = normalize_plate(plate)
    if not plate_norm or len(plate_norm) < 3:
        return []

    results = []
    with open_conn() as conn:
        rows = conn.execute(
            """SELECT *, 'log' AS source FROM plates_log
               WHERE plate_norm = ?
               ORDER BY id DESC""",
            (plate_norm,)
        ).fetchall()
        results.extend([dict(r) for r in rows])

        rows = conn.execute(
            """SELECT plate_norm, plate_raw, added_at, photo_full, photo_crop,
                      'allowed' AS source
               FROM plates_allowed WHERE plate_norm = ?""",
            (plate_norm,)
        ).fetchall()
        results.extend([dict(r) for r in rows])

        rows = conn.execute(
            """SELECT plate_norm, plate_raw, added_at, reason, active,
                      photo_full, photo_crop, 'watch' AS source
               FROM plates_watch WHERE plate_norm = ?""",
            (plate_norm,)
        ).fetchall()
        results.extend([dict(r) for r in rows])
    return results


def get_log_stats() -> Dict[str, Any]:
    today_str = datetime.now().strftime("%Y-%m-%d")
    with open_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM plates_log").fetchone()["c"]
        unique = conn.execute(
            "SELECT COUNT(DISTINCT plate_norm) AS c FROM plates_log"
        ).fetchone()["c"]
        today = conn.execute(
            "SELECT COUNT(*) AS c FROM plates_log WHERE timestamp LIKE ?",
            (f"{today_str}%",)
        ).fetchone()["c"]

        unreviewed = 0
        try:
            unreviewed = conn.execute(
                "SELECT COUNT(*) AS c FROM plates_log WHERE reviewed = 0"
            ).fetchone()["c"]
        except sqlite3.DatabaseError:
            pass

    return {"total": total, "unique": unique, "today": today,
            "unreviewed": unreviewed}


# -----------------------------------------------------------------------------
# БАЗА 2 — РАЗРЕШЁННЫЕ НОМЕРА
# -----------------------------------------------------------------------------
def add_plate(plate: str, status: str = "allowed"):
    ts = datetime.now().isoformat()
    plate_norm = normalize_plate(plate)
    with open_conn() as conn:
        if status == "allowed":
            conn.execute(
                """INSERT OR REPLACE INTO plates_allowed
                (plate_norm, plate_raw, added_at, notes)
                VALUES (?, ?, ?, ?)""",
                (plate_norm, plate, ts, None),
            )
        else:
            conn.execute(
                """INSERT OR REPLACE INTO plates_watch
                (plate_norm, plate_raw, added_at, reason, active)
                VALUES (?, ?, ?, ?, 1)""",
                (plate_norm, plate, ts, "manual"),
            )
        conn.commit()


def delete_plate_with_photos(plate: str):
    plate_norm = normalize_plate(plate)
    with open_conn() as conn:
        rows_allowed = conn.execute(
            "SELECT photo_full, photo_crop FROM plates_allowed WHERE plate_norm = ?",
            (plate_norm,)
        ).fetchall()
        rows_watch = conn.execute(
            "SELECT photo_full, photo_crop FROM plates_watch WHERE plate_norm = ?",
            (plate_norm,)
        ).fetchall()
        paths = _collect_photos_from_rows(rows_allowed) + _collect_photos_from_rows(rows_watch)

        conn.execute("DELETE FROM plates_allowed WHERE plate_norm = ?", (plate_norm,))
        conn.execute("DELETE FROM plates_watch WHERE plate_norm = ?", (plate_norm,))
        conn.commit()

    _delete_photo_files(paths)


def is_allowed_plate(plate_norm: str) -> Optional[Dict[str, Any]]:
    plate_norm = normalize_plate(plate_norm)
    with open_conn() as conn:
        r = conn.execute(
            """SELECT plate_norm, plate_raw, added_at, photo_full, photo_crop
               FROM plates_allowed WHERE plate_norm = ? LIMIT 1""",
            (plate_norm,),
        ).fetchone()
        return dict(r) if r else None


def get_allowed_list() -> List[Dict[str, Any]]:
    with open_conn() as conn:
        rows = conn.execute(
            """SELECT plate_norm, plate_raw, added_at, photo_full, photo_crop
               FROM plates_allowed ORDER BY added_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


# -----------------------------------------------------------------------------
# БАЗА 3 — СПИСОК КОНТРОЛЯ
# -----------------------------------------------------------------------------
def add_watch_plate(plate_raw: str, plate_norm: str, reason: str) -> bool:
    try:
        ts = datetime.now().isoformat()
        norm = normalize_plate(plate_norm) if plate_norm else normalize_plate(plate_raw)
        if not norm:
            norm = normalize_plate(plate_raw)
        with open_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO plates_watch
                (plate_norm, plate_raw, added_at, reason, active)
                VALUES (?, ?, ?, ?, 1)""",
                (norm, plate_raw, ts, reason),
            )
            conn.commit()
        return True
    except sqlite3.DatabaseError as e:
        logger.exception("add_watch_plate failed: %s", e)
        return False


def deactivate_watch_plate(plate_norm: str) -> bool:
    plate_norm = normalize_plate(plate_norm)
    with open_conn() as conn:
        res = conn.execute(
            "UPDATE plates_watch SET active = 0 WHERE plate_norm = ?",
            (plate_norm,),
        )
        conn.commit()
        return res.rowcount > 0


def is_watch_plate(plate_norm: str) -> Optional[Dict[str, Any]]:
    plate_norm = normalize_plate(plate_norm)
    with open_conn() as conn:
        r = conn.execute(
            """SELECT plate_norm, plate_raw, added_at, reason, active,
                      photo_full, photo_crop
               FROM plates_watch WHERE plate_norm = ? AND active = 1 LIMIT 1""",
            (plate_norm,),
        ).fetchone()
        return dict(r) if r else None


def get_watch_list() -> List[Dict[str, Any]]:
    with open_conn() as conn:
        rows = conn.execute(
            """SELECT plate_norm, plate_raw, added_at, reason, active,
                      photo_full, photo_crop
               FROM plates_watch ORDER BY added_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


# -----------------------------------------------------------------------------
# ЖУРНАЛ ДЛЯ ЭКРАНА «ПРОВЕРКА»
# -----------------------------------------------------------------------------
def get_log_list(limit: int = 100, unreviewed_only: bool = False) -> List[Dict[str, Any]]:
    with open_conn() as conn:
        if unreviewed_only:
            rows = conn.execute(
                """SELECT id, plate_raw, plate_norm, timestamp, frame_path,
                   plate_crop_path, yolo_conf, ocr_score, winner, reviewed,
                   mode, barrier_opened
                   FROM plates_log
                   WHERE reviewed = 0
                   ORDER BY id DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, plate_raw, plate_norm, timestamp, frame_path,
                   plate_crop_path, yolo_conf, ocr_score, winner, reviewed,
                   mode, barrier_opened
                   FROM plates_log
                   ORDER BY id DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def update_plate_reviewed(log_id: int, plate_raw: str = None, plate_norm: str = None) -> bool:
    try:
        with open_conn() as conn:
            if plate_raw is not None and plate_norm is not None:
                conn.execute(
                    """UPDATE plates_log
                       SET reviewed = 1, plate_raw = ?, plate_norm = ?
                       WHERE id = ?""",
                    (plate_raw, plate_norm, log_id)
                )
            else:
                conn.execute(
                    "UPDATE plates_log SET reviewed = 1 WHERE id = ?",
                    (log_id,)
                )
            conn.commit()
        return True
    except sqlite3.DatabaseError as e:
        logger.exception("update_plate_reviewed failed: %s", e)
        return False


def get_unreviewed_count() -> int:
    with open_conn() as conn:
        try:
            return conn.execute(
                "SELECT COUNT(*) AS c FROM plates_log WHERE reviewed = 0"
            ).fetchone()["c"]
        except sqlite3.DatabaseError:
            return 0

