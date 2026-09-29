"""Общие утилиты проекта: нормализация номеров, транслитерация, валидация формата."""

import re

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
    """Нормализует номер: верхний регистр, кириллица → латиница,
    только буквы и цифры (убирает пробелы, дефисы, точки и прочий мусор)."""
    if not plate:
        return ""
    plate = plate.upper().strip()
    result = []
    for ch in plate:
        if ch in CYR_TO_LAT:
            result.append(CYR_TO_LAT[ch])
        elif ch.isalnum():
            result.append(ch)
    return ''.join(result)


def to_cyrillic(plate: str) -> str:
    """Меняет латинские буквы на похожие русские для отображения на экране."""
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
    """Проверяет, соответствует ли номер формату российского госномера."""
    if not plate:
        return False
    return bool(RUSSIAN_PLATE_PATTERN.match(plate))