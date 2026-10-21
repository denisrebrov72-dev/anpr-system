"""Общие утилиты проекта: нормализация номеров, транслитерация, валидация формата."""
import re
# 12 букв, используемых на российских госномерах.
# Все они визуально совпадают с латинскими — это позволяет OCR работать
# на латинице, а оператору показывать кириллицу через to_cyrillic().
PLATE_LETTERS = "ABCEHKMOPTXY"

CYR_TO_LAT = {
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H',
    'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'У': 'Y', 'Х': 'X',
}

LAT_TO_CYR = {
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М', 'H': 'Н',
    'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т', 'X': 'Х', 'Y': 'У',
}

RUSSIAN_PLATE_PATTERN = re.compile(
    rf'^[{PLATE_LETTERS}]\d{{3}}[{PLATE_LETTERS}]{{2}}\d{{2,3}}$'
)

def normalize_plate(plate: str) -> str:
    """Нормализует номер к единому виду для поиска и сравнения:
       - верхний регистр;
       - кириллица → латиница (для 12 «двойных» букв);
       - оставляем только разрешённые символы: 12 латинских букв + цифры;
       - пробелы, дефисы, точки и прочий мусор — выбрасываем.

    Примеры:
        'а 123 вс 77'   -> 'A123BC77'
        'А-123-ВС-77'   -> 'A123BC77'
        'A123BC777'     -> 'A123BC777'
    """
    if not plate:
        return ""
    plate = plate.upper().strip()
    result = []
    for ch in plate:
        if ch in CYR_TO_LAT:
            result.append(CYR_TO_LAT[ch])
        elif ch in PLATE_LETTERS or ch.isdigit():
            result.append(ch)
        # всё остальное (пробелы, дефисы, кириллица вне набора, спецсимволы) — отбрасываем
    return ''.join(result)


def to_cyrillic(plate: str) -> str:
    """Меняет латинские буквы на похожие русские для отображения на экране.
    В базе по-прежнему хранится латиница — это только представление."""
    if not plate:
        return ""
    result = []
    for ch in plate.upper():
        if ch in LAT_TO_CYR:
            result.append(LAT_TO_CYR[ch])
        else:
            result.append(ch)
    return ''.join(result)


def is_valid_russian_plate(plate: str) -> bool:
    """Проверяет, соответствует ли номер формату российского госномера
    (стандартный легковой: А123ВС77 / А123ВС777).

    Мотоциклы, прицепы, такси и спецтранспорт имеют другие форматы —
    они попадают в needs_review и уточняются оператором вручную.
    """
    if not plate:
        return False
    return bool(RUSSIAN_PLATE_PATTERN.match(plate))