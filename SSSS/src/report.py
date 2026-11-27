"""Сборка итогового отчёта в Markdown (на русском языке).
Отчёт собирается кодом из проверенных данных, поэтому все цитаты и места — из документа."""

import subprocess
from collections import Counter
from datetime import datetime

from src import AGENT_VERSION
from src.document import Document

SEVERITY_ORDER = {"критично": 0, "существенно": 1, "незначительно": 2}


def _cell(text: str) -> str:
    """Безопасный текст для ячейки Markdown-таблицы."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _quote(text: str) -> str:
    return "«" + " ".join(text.split()) + "»"


def _agent_version() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return f"{AGENT_VERSION} (коммит {commit})" if commit else AGENT_VERSION


def build_report(doc: Document, model: str, reqs: list[dict], set_result: dict, stats: dict) -> str:
    set_issues = sorted(set_result["issues"], key=lambda i: SEVERITY_ORDER[i["severity"]])
    req_issues = [(r, i) for r in reqs for i in r["issues"]]
    all_sev = Counter(i["severity"] for _, i in req_issues) + Counter(i["severity"] for i in set_issues)
    total = len(req_issues) + len(set_issues)
    out: list[str] = []
    w = out.append

    w("# Отчёт о проверке требований\n")

    w("## 1. Паспорт прогона\n")
    w("| Поле | Значение |\n| :--- | :--- |")
    w(f"| Документ | {_cell(doc.title)} |")
    w(f"| Файл | `{_cell(doc.path.name)}` |")
    w(f"| Объём | {len(doc.segments)} {'стр.' if doc.kind == 'pdf' else 'строк'} |")
    w(f"| Дата прогона | {datetime.now():%Y-%m-%d %H:%M} |")
    w(f"| Модель | {model} |")
    w(f"| Версия агента | {_agent_version()} |")
    w("| Критерии | ISO/IEC/IEEE 29148:2018 (п. 5.2.5, 5.2.6), IEEE Std 830-1998 (п. 4.3) |\n")

    w("## 2. Таблица нарезки требований\n")
    w("Если в документе нет номера требования, указано «—»; используется идентификатор отчёта и цитата.\n")
    w("| ID в отчёте | ID в документе | Раздел | Место | Краткая формулировка |")
    w("| :--- | :--- | :--- | :--- | :--- |")
    for r in reqs:
        w(f"| {r['id']} | {_cell(r['source_id'])} | {_cell(r['section'])} | {r['location']} | {_cell(r['summary'])} |")
    w("")

    w("## 3. Проверка каждого требования\n")
    for r in reqs:
        title = f"{r['id']}" + (f" ({r['source_id']})" if r["source_id"] != "—" else "")
        w(f"### {title}: {r['summary']}\n")
        w(f"> {_quote(r['quote'])} — *{r['location']}*\n")
        if not r["issues"]:
            w("Нарушений не выявлено.\n")
            continue
        for n, i in enumerate(sorted(r["issues"], key=lambda x: SEVERITY_ORDER[x["severity"]]), start=1):
            w(f"{n}. **[{i['severity']}] {i['criterion']}.** {i['problem']}")
            w(f"   - Цитата: {_quote(i['quote'])} ({i['location']})")
            w(f"   - Рекомендация: {i['recommendation']}")
        w("")

    w("## 4. Проверка всего набора требований\n")
    if not set_issues:
        w("Замечаний к набору в целом не выявлено.\n")
    for n, i in enumerate(set_issues, start=1):
        w(f"### S-{n:02d}. [{i['severity']}] {i['category']}\n")
        w(i["problem"] + "\n")
        for e in i["evidence"]:
            w(f"- {_quote(e['quote'])} ({e['location']})")
        w(f"\n**Рекомендация:** {i['recommendation']}\n")

    w("## 5. Сводка\n")
    w("| Показатель | Значение |\n| :--- | :--- |")
    w(f"| Требований выделено | {len(reqs)} |")
    w(f"| Требований с нарушениями | {sum(1 for r in reqs if r['issues'])} |")
    w(f"| Нарушений всего | {total} (по требованиям: {len(req_issues)}, по набору: {len(set_issues)}) |")
    for s in SEVERITY_ORDER:
        w(f"| — {s} | {all_sev.get(s, 0)} |")
    w("")
    criteria = Counter(i["criterion"] for _, i in req_issues)
    if criteria:
        w("**Нарушения по критериям:** " + "; ".join(f"{c} — {n}" for c, n in criteria.most_common()) + "\n")

    w("### Критичные дефекты\n")
    critical = [(f"{r['id']} ({r['source_id']})", i["criterion"], i["problem"], i["location"])
                for r, i in req_issues if i["severity"] == "критично"]
    critical += [(f"S-{n:02d}", i["category"], i["problem"], i["evidence"][0]["location"])
                 for n, i in enumerate(set_issues, start=1) if i["severity"] == "критично"]
    if critical:
        w("| Где | Критерий | Дефект | Место |\n| :--- | :--- | :--- | :--- |")
        for where, crit, problem, loc in critical:
            w(f"| {_cell(where)} | {_cell(crit)} | {_cell(problem)} | {loc} |")
    else:
        w("Критичных дефектов не выявлено.")
    w("")

    w("## 6. Итог\n")
    ready = set_result["ready"] and not critical
    w(f"**Можно ли начинать разработку и тестирование:** {'да' if ready else 'нет, сначала нужно исправить документ'}.\n")
    if set_result["conclusion"]:
        w(set_result["conclusion"] + "\n")
    if set_result["top_fixes"]:
        w("**Исправить в первую очередь:**\n")
        for n, fix in enumerate(set_result["top_fixes"], start=1):
            w(f"{n}. {fix}")
        w("")

    w("---\n")
    w("*Контроль доказательств:* каждая цитата в отчёте найдена в исходном документе автоматически, "
      "место (страница/строка) вычислено по тексту документа. "
      f"Отброшено требований с ненайденной цитатой: {stats['dropped_requirements']}; "
      f"замечаний к набору без подтверждённой цитаты: {stats['dropped_set_issues']}; "
      f"замечаний, привязанных к тексту требования вместо неточной цитаты: {stats['requotes']}.")
    return "\n".join(out) + "\n"
