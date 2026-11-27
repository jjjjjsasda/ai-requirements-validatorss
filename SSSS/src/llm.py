"""Обёртка над Gemini API: JSON-ответ по схеме Pydantic, повторы при ошибках."""

import json
import os
import re
import time
from typing import TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK = ("gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-2.5-pro,"
                    "gemini-3.5-flash-lite,gemini-3.1-flash-lite")
MAX_ATTEMPTS = 4

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Модель недоступна или вернула некорректный ответ."""


class GeminiClient:
    def __init__(self, model_name: str | None = None):
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise LLMError("Не задан GEMINI_API_KEY. Скопируйте .env.example в .env и укажите ключ.")
        # если в GEMINI_MODEL по ошибке указан список через запятую — первая модель основная, остальные резервные
        names = [m.strip() for m in (model_name or os.getenv("GEMINI_MODEL", "")).split(",") if m.strip()]
        self.model = names[0] if names else DEFAULT_MODEL
        fallback = os.getenv("GEMINI_FALLBACK_MODELS", "").strip() or DEFAULT_FALLBACK
        self.fallbacks = []
        for m in names[1:] + fallback.split(","):
            m = m.strip()
            if m and m != self.model and m not in self.fallbacks:
                self.fallbacks.append(m)
        self.used_models: list[str] = []  # какие модели реально ответили (для паспорта отчёта)
        self.exhausted: set[str] = set()  # модели с исчерпанной дневной квотой или 404
        # таймаут одного запроса — 5 минут (в миллисекундах)
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=300_000))

    def ask_json(self, system: str, prompt: str, schema: type[T]) -> T:
        """Как _ask_one, но при перегрузке/лимите основной модели пробует резервные."""
        for round_no in (1, 2):  # второй круг — после паузы, если все модели были заняты
            candidates = [self.model] + self.fallbacks
            # начинаем с последней модели, которая ответила, чтобы не тратить лимит на перегруженную
            if self.used_models:
                last = self.used_models[-1]
                candidates = [last] + [m for m in candidates if m != last]
            candidates = [m for m in candidates if m not in self.exhausted]
            for model in candidates:
                try:
                    result = self._ask_one(model, system, prompt, schema)
                except _Overloaded as exc:
                    print(f"  модель {model} недоступна ({exc}), пробуем следующую")
                    continue
                if model in self.used_models:
                    self.used_models.remove(model)
                self.used_models.append(model)
                return result
            if round_no == 1 and candidates:
                print("  все модели заняты, ждём 90 с и пробуем ещё раз")
                time.sleep(90)
        raise LLMError("Все модели перегружены или исчерпан лимит запросов. Попробуйте позже.")

    def _ask_one(self, model: str, system: str, prompt: str, schema: type[T]) -> T:
        """Отправляет запрос и возвращает ответ, провалидированный по схеме.
        Повторяет запрос при сетевых ошибках, лимитах (429/5xx) и битом JSON."""
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=schema,
        )
        last_error, overloaded = "", False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            overloaded = False
            try:
                response = self.client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                return _parse(response, schema)
            except errors.ClientError as exc:
                code = getattr(exc, "code", None)
                if code == 429 and "PerDay" in str(exc):  # дневная квота — ждать бессмысленно
                    self.exhausted.add(model)
                    raise _Overloaded("исчерпана дневная квота (429)") from exc
                if code == 429:  # минутный лимит бесплатного тарифа
                    overloaded = True
                    last_error = "превышен лимит запросов (429)"
                    wait = min(_retry_delay(str(exc)) or 20 * attempt, 60)
                elif code == 404 and self.fallbacks:
                    self.exhausted.add(model)
                    raise _Overloaded("модель не найдена (404)") from exc
                elif code in (400, 401, 403, 404):
                    raise LLMError(f"Ошибка запроса к модели «{model}» ({code}): {exc}") from exc
                else:
                    last_error, wait = str(exc), 5 * attempt
            except errors.ServerError as exc:
                overloaded = True
                last_error, wait = f"ошибка сервера {getattr(exc, 'code', '')}", 10 * attempt
            except (ValidationError, ValueError) as exc:
                last_error, wait = f"некорректный JSON от модели: {exc}", 2
            except Exception as exc:  # сеть, таймауты — ждём дольше, смена модели не поможет
                last_error, wait = f"сетевая ошибка {type(exc).__name__}: {exc}", 15 * attempt
            if attempt < MAX_ATTEMPTS:
                print(f"  [{model}, повтор {attempt}/{MAX_ATTEMPTS - 1}] {last_error[:200]}; ждём {wait} с")
                time.sleep(wait)
        if overloaded:
            raise _Overloaded(last_error)
        raise LLMError(f"Модель «{model}» не ответила после {MAX_ATTEMPTS} попыток: {last_error}")


class _Overloaded(Exception):
    """Модель перегружена/лимит — можно переключиться на резервную."""


def _parse(response, schema: type[T]) -> T:
    if isinstance(getattr(response, "parsed", None), schema):
        return response.parsed
    text = (response.text or "").strip()
    if not text:
        raise ValueError("пустой ответ (возможно, сработал фильтр или лимит токенов)")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return schema.model_validate(json.loads(text))


def _retry_delay(message: str) -> int | None:
    match = re.search(r"retry in ([\d.]+)s|retryDelay['\"]?: ?['\"](\d+)s", message)
    if not match:
        return None
    return int(float(match.group(1) or match.group(2))) + 2
