"""Пайплайн проверки: нарезка требований -> проверка каждого -> проверка набора -> отчёт."""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.document import Document, load_document, locate_quote, normalize
from src.llm import GeminiClient
from src.report import build_report

EXTRACT_CHUNK_CHARS = 25_000   # размер куска документа для нарезки
ANALYSIS_BATCH = 20            # требований за один запрос проверки (меньше запросов — меньше 429)
FULL_CONTEXT_CHARS = 150_000   # до этого размера модель видит документ целиком

SEVERITIES = ("критично", "существенно", "незначительно")

CRITERIA = """Критерии отдельного требования (ISO/IEC/IEEE 29148:2018 п. 5.2.5; IEEE 830-1998 п. 4.3):
- Необходимость; Однозначность (одно толкование, нет размытых слов: «быстро», «удобно», «и т.д.», «при необходимости»);
- Полнота (нет недосказанности: что при ошибке, граничные значения, единицы, роли); Единичность/атомарность;
- Осуществимость; Проверяемость (можно написать тест с ожидаемым результатом); Корректность;
- Независимость от реализации (нет лишних проектных решений); Прослеживаемость (есть идентификатор, источник);
- Соответствие шаблону/терминологии.
Критерии набора требований (ISO/IEC/IEEE 29148:2018 п. 5.2.6; IEEE 830-1998 п. 4.3):
- Полнота набора (пропущенные сценарии, ошибки, роли, состояния, нефункциональные требования, пустые разделы);
- Согласованность (противоречия между разделами, единицы измерения, коды ответов, методы API);
- Отсутствие дублирования; Единая терминология; Осуществимость набора; Понятность;
- Ранжирование по важности/стабильности; Модифицируемость; Прослеживаемость; Версионность документа."""

RULES = """Жёсткие правила:
1. Поле quote — ТОЧНАЯ дословная выдержка из документа (копируй символы как есть, без пересказа,
   без многоточий, без маркеров [N] и «=== Страница ===»). Лучше короткая точная цитата (5–25 слов),
   чем длинная неточная. Цитаты, которых нет в документе, будут отброшены.
   Цитата должна быть именно тем текстом, о котором замечание (если дефект в другом месте документа —
   цитируй то место, а не текст проверяемого требования).
2. Не выдумывай факты. Если замечание нельзя подтвердить цитатой — не пиши его.
3. Все тексты — на русском языке. Серьёзность: «критично», «существенно» или «незначительно».
   «критично» — непроверяемое, противоречивое или неполное требование, мешающее разработке/тестированию."""


# ---------- Схемы ответов модели ----------

class ExtractedRequirement(BaseModel):
    source_id: str = Field(description="Идентификатор в документе (например «Ф-1»), или пустая строка")
    section: str = Field(description="Раздел документа, где находится требование")
    quote: str = Field(description="Дословный текст требования из документа")
    summary: str = Field(description="Краткая формулировка требования, до 15 слов")


class ExtractionResult(BaseModel):
    requirements: list[ExtractedRequirement]


class Issue(BaseModel):
    criterion: str = Field(description="Нарушенный критерий качества")
    severity: str = Field(description="критично | существенно | незначительно")
    quote: str = Field(description="Дословная цитата из документа, подтверждающая замечание")
    problem: str = Field(description="В чём дефект")
    recommendation: str = Field(description="Как исправить")


class RequirementAnalysis(BaseModel):
    req_id: str
    issues: list[Issue]


class AnalysisBatch(BaseModel):
    results: list[RequirementAnalysis]


class SetIssue(BaseModel):
    category: str = Field(description="Противоречие | Дублирование | Пропуск | Терминология | Согласованность | Версионность | Прочее")
    severity: str
    quotes: list[str] = Field(description="Одна или несколько дословных цитат из документа")
    problem: str
    recommendation: str


class SetAnalysis(BaseModel):
    issues: list[SetIssue]
    ready_for_development: bool
    conclusion: str = Field(description="Итог: можно ли начинать разработку и тестирование, 2–4 предложения")
    top_fixes: list[str] = Field(description="Что исправить в первую очередь, 3–7 пунктов")


# ---------- Этапы пайплайна ----------

def extract_requirements(llm: GeminiClient, doc: Document, stats: dict) -> list[dict]:
    system = (
        "Ты — аналитик требований. Выдели из фрагмента документа все отдельные (атомарные) требования: "
        "функциональные, нефункциональные, к данным, к интеграции, к API (параметры, статусы, правила), "
        "критерии приёмки. Составное требование раздели на части, но quote каждой части должен быть "
        "дословным куском текста. Будь исчерпывающим, отдельными требованиями считаются: каждое "
        "предложение-требование в тексте; каждая строка таблицы критериев приёмки; каждый параметр запроса "
        "и ответа API (строка таблицы); каждый HTTP-статус с его условием; каждое правило и шаг логики метода. "
        "Для строки таблицы quote — текст ячеек строки через пробел, без символов «|». "
        "Заголовки, чек-листы ревью, пустые шаблоны и подписи не являются требованиями.\n"
        + RULES
    )
    chunks = doc.chunks(EXTRACT_CHUNK_CHARS)
    found, seen = [], set()
    for i, chunk in enumerate(chunks, start=1):
        print(f"[1/3] Нарезка требований: фрагмент {i}/{len(chunks)}")
        result = llm.ask_json(system, f"Фрагмент документа:\n\n{chunk}", ExtractionResult)
        for req in result.requirements:
            key = normalize(req.quote)
            if not key or key in seen:
                continue
            location = locate_quote(doc, req.quote)
            if location is None:
                stats["dropped_requirements"] += 1
                continue
            seen.add(key)
            found.append({
                "id": f"R-{len(found) + 1:03d}",
                "source_id": req.source_id.strip() or "—",
                "section": req.section.strip(),
                "quote": req.quote.strip(),
                "summary": req.summary.strip(),
                "location": location,
                "issues": [],
            })
    return found


def _context(doc: Document) -> str:
    text = doc.numbered_text()
    if len(text) <= FULL_CONTEXT_CHARS:
        return f"Полный текст документа (для контекста):\n\n{text}\n\n"
    return ""  # очень большой документ: проверяем только по тексту требований


def _severity(value: str) -> str:
    value = value.strip().lower()
    for s in SEVERITIES:
        if value.startswith(s[:5]):
            return s
    return "существенно"


CRITERION_NAMES = {
    "неоднознач": "Однозначность", "однознач": "Однозначность", "полнот": "Полнота",
    "неполн": "Полнота", "проверяем": "Проверяемость", "верифиц": "Проверяемость",
    "непроверяем": "Проверяемость", "согласован": "Согласованность", "противореч": "Согласованность",
    "единичн": "Единичность", "атомарн": "Единичность", "осуществим": "Осуществимость",
    "прослежив": "Прослеживаемость", "необходим": "Необходимость", "корректн": "Корректность",
    "независим": "Независимость от реализации", "реализац": "Независимость от реализации",
    "соответств": "Соответствие", "терминолог": "Терминология",
}


def _criterion(value: str) -> str:
    """Приводит название критерия к единому виду, чтобы сводка не дробилась."""
    low = value.strip().lower()
    for stem, name in CRITERION_NAMES.items():
        if low.startswith(stem):
            return name
    return value.strip() or "Прочее"


def analyze_requirements(llm: GeminiClient, doc: Document, reqs: list[dict], stats: dict) -> None:
    system = (
        "Ты — эксперт по качеству требований. Проверь КАЖДОЕ переданное требование по критериям ниже. "
        "Для каждого req_id верни список замечаний (пустой, если требование качественное). "
        "Не дублируй одно и то же замечание у разных требований без причины.\n"
        + CRITERIA + "\n" + RULES
    )
    context = _context(doc)
    by_id = {r["id"]: r for r in reqs}
    batches = [reqs[i:i + ANALYSIS_BATCH] for i in range(0, len(reqs), ANALYSIS_BATCH)]
    for n, batch in enumerate(batches, start=1):
        print(f"[2/3] Проверка требований: пакет {n}/{len(batches)}")
        payload = [{"req_id": r["id"], "source_id": r["source_id"], "section": r["section"],
                    "text": r["quote"]} for r in batch]
        prompt = context + "Требования для проверки:\n" + json.dumps(payload, ensure_ascii=False, indent=1)
        result = llm.ask_json(system, prompt, AnalysisBatch)
        for item in result.results:
            req_id = item.req_id.strip().upper()
            if req_id.startswith("R-") and req_id[2:].isdigit():
                req_id = f"R-{int(req_id[2:]):03d}"  # модель могла вернуть «R-1» вместо «R-001»
            req = by_id.get(req_id)
            if req is None:
                continue
            for issue in item.issues:
                location = locate_quote(doc, issue.quote)
                quote = issue.quote.strip()
                if location is None:
                    # цитата модели не найдена — привязываем замечание к тексту самого требования
                    stats["requotes"] += 1
                    quote, location = req["quote"], req["location"]
                req["issues"].append({
                    "criterion": _criterion(issue.criterion),
                    "severity": _severity(issue.severity),
                    "quote": quote,
                    "location": location,
                    "problem": issue.problem.strip(),
                    "recommendation": issue.recommendation.strip(),
                })


def analyze_set(llm: GeminiClient, doc: Document, reqs: list[dict], stats: dict) -> dict:
    print("[3/3] Проверка набора требований целиком")
    system = (
        "Ты — эксперт по качеству спецификаций. Проверь набор требований ЦЕЛИКОМ: противоречия между "
        "разделами и частями документа, дублирование, пропущенные требования и сценарии, пустые/незаполненные "
        "разделы, расхождения терминологии и единиц измерения, несогласованность API (методы, коды ответов, "
        "типы полей) с функциональными требованиями, версионность. Для противоречий приводи цитаты обеих сторон. "
        "Для пропусков цитируй место, где требование должно быть или на которое оно опирается. "
        "Также дай итог: можно ли по документу начинать разработку и тестирование.\n"
        + CRITERIA + "\n" + RULES
    )
    req_list = [{"req_id": r["id"], "source_id": r["source_id"], "text": r["quote"]} for r in reqs]
    prompt = (
        _context(doc)
        + "Выделенные требования:\n" + json.dumps(req_list, ensure_ascii=False, indent=1)
    )
    result = llm.ask_json(system, prompt, SetAnalysis)

    issues = []
    for issue in result.issues:
        evidence = []
        for q in issue.quotes:
            location = locate_quote(doc, q)
            if location:
                evidence.append({"quote": q.strip(), "location": location})
        if not evidence:
            stats["dropped_set_issues"] += 1  # нет ни одной подтверждённой цитаты — не включаем
            continue
        issues.append({
            "category": issue.category.strip() or "Прочее",
            "severity": _severity(issue.severity),
            "evidence": evidence,
            "problem": issue.problem.strip(),
            "recommendation": issue.recommendation.strip(),
        })
    return {
        "issues": issues,
        "ready": result.ready_for_development,
        "conclusion": result.conclusion.strip(),
        "top_fixes": [f.strip() for f in result.top_fixes if f.strip()],
    }


def run_validation(input_path: Path, output_path: Path, model_name: str | None = None) -> None:
    doc = load_document(input_path)
    print(f"Документ: {doc.path.name} ({len(doc.segments)} {'стр.' if doc.kind == 'pdf' else 'строк'})")
    llm = GeminiClient(model_name)
    print(f"Модель: {llm.model}")

    stats = {"dropped_requirements": 0, "requotes": 0, "dropped_set_issues": 0}
    reqs = extract_requirements(llm, doc, stats)
    if not reqs:
        raise RuntimeError("Модель не выделила ни одного требования с подтверждённой цитатой.")
    print(f"  выделено требований: {len(reqs)}")
    analyze_requirements(llm, doc, reqs, stats)
    set_result = analyze_set(llm, doc, reqs, stats)

    report = build_report(doc, ", ".join(llm.used_models), reqs, set_result, stats)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Отчёт сохранён: {output_path.resolve()}")
