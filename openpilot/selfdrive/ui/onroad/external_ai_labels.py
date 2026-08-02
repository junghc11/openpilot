from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from openpilot.selfdrive.ui.translations.potools import parse_po


def tr_noop(text: str) -> str:
  """Mark a UI string for the translation extractor without loading UI state."""
  return text


def _identity(text: str) -> str:
  return text


DISPLAY_NAME_KEYS = {
  "car": tr_noop("Car"),
  "truck": tr_noop("Truck"),
  "bus": tr_noop("Bus"),
  "motorcycle": tr_noop("Motorcycle"),
  "bicycle": tr_noop("Bicycle"),
  "person": tr_noop("Pedestrian"),
  "traffic light": tr_noop("Traffic Light"),
  "stop sign": tr_noop("Stop Sign"),
}

SUPPORTED_DISPLAY_LANGUAGES = frozenset((
  "en", "de", "fr", "pt-BR", "es", "tr", "uk", "th", "zh-CHT", "zh-CHS", "ko", "ja",
))
TRANSLATIONS_DIR = Path(__file__).resolve().parents[1] / "translations"


def external_ai_display_name(class_name: str, translate: Callable[[str], str] = _identity) -> str:
  normalized = str(class_name or "").strip().lower()
  translation_key = DISPLAY_NAME_KEYS.get(normalized)
  if translation_key is None:
    return normalized.upper()
  return translate(translation_key) or translation_key


def normalize_external_ai_language(language: object) -> str:
  if isinstance(language, bytes):
    language = language.decode("utf-8", "ignore")
  normalized = str(language or "").strip().removeprefix("main_")
  return normalized if normalized in SUPPORTED_DISPLAY_LANGUAGES else "en"


@lru_cache(maxsize=len(SUPPORTED_DISPLAY_LANGUAGES))
def _translations_for_language(language: str) -> dict[str, str]:
  normalized = normalize_external_ai_language(language)
  try:
    _, entries = parse_po(TRANSLATIONS_DIR / f"app_{normalized}.po")
  except (OSError, ValueError):
    return {}
  return {entry.msgid: entry.msgstr for entry in entries if entry.msgstr}


def external_ai_display_name_for_language(class_name: str, language: object) -> str:
  translations = _translations_for_language(normalize_external_ai_language(language))
  return external_ai_display_name(class_name, translate=lambda text: translations.get(text, text))
