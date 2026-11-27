"""Чтение входного документа и поиск цитат в исходном тексте.

Документ хранится как список сегментов: для PDF сегмент = страница,
для .md/.txt сегмент = строка. Каждая цитата, которую вернула модель,
проверяется по исходному тексту, а её местоположение вычисляется кодом,
а не берётся со слов модели.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


class DocumentError(Exception):
    """Ошибка чтения входного документа."""


@dataclass
class Segment:
    number: int  # номер страницы (PDF) или строки (md/txt), с 1
    text: str


@dataclass
class Document:
    path: Path
    kind: str  # "pdf" или "text"
    segments: list[Segment]

    @property
    def title(self) -> str:
        """Первый заголовок/непустая строка документа или имя файла."""
        for seg in self.segments:
            for line in seg.text.splitlines():
                line = line.strip().lstrip("#").strip()
                if line:
                    return line[:150]
        return self.path.name

    @property
    def unit_name(self) -> str:
        return "стр." if self.kind == "pdf" else "строка"

    def numbered_text(self) -> str:
        """Текст с маркерами страниц/строк — так модель видит структуру документа."""
        if self.kind == "pdf":
            return "\n\n".join(f"=== Страница {s.number} ===\n{s.text}" for s in self.segments)
        return "\n".join(f"[{s.number}] {s.text}" for s in self.segments)

    def chunks(self, max_chars: int) -> list[str]:
        """Делит документ на куски не длиннее max_chars по границам сегментов."""
        result, current, size = [], [], 0
        for seg in self.segments:
            piece = (
                f"=== Страница {seg.number} ===\n{seg.text}"
                if self.kind == "pdf"
                else f"[{seg.number}] {seg.text}"
            )
            if current and size + len(piece) > max_chars:
                result.append("\n".join(current))
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 1
        if current:
            result.append("\n".join(current))
        return result


def load_document(path: Path) -> Document:
    if not path.exists():
        raise DocumentError(f"Файл не найден: {path}")
    if not path.is_file():
        raise DocumentError(f"Путь не является файлом: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentError(
            f"Неподдерживаемый формат «{ext}». Поддерживаются: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                reader.decrypt("")
            segments = [
                Segment(i, (page.extract_text() or "").strip())
                for i, page in enumerate(reader.pages, start=1)
            ]
        except (PdfReadError, OSError, ValueError) as exc:
            raise DocumentError(f"Не удалось прочитать PDF «{path.name}»: {exc}") from exc
        doc = Document(path, "pdf", segments)
    else:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="cp1251")
        segments = [Segment(i, line.rstrip()) for i, line in enumerate(text.splitlines(), start=1)]
        doc = Document(path, "text", segments)

    if not any(s.text.strip() for s in doc.segments):
        raise DocumentError(
            "В документе не найден текст. Если это скан PDF, нужен PDF с текстовым слоем."
        )
    return doc


# ---------- Проверка цитат ----------

_STRIP_CHARS = re.compile(r"[*_`#|>\\]")
_QUOTES = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "'": '"',
                         "–": "-", "—": "-", "−": "-", "ё": "е", " ": " ", " ": " "})


def normalize(text: str) -> str:
    """Убирает разметку и различия в пробелах/кавычках/тире, чтобы сравнивать текст."""
    text = text.lower().translate(_QUOTES)
    text = _STRIP_CHARS.sub(" ", text)
    text = re.sub(r"-\s*\n\s*", "", text)  # перенос слова в PDF
    return re.sub(r"\s+", " ", text).strip()


def locate_quote(doc: Document, quote: str, max_span: int = 6) -> str | None:
    """Возвращает местоположение цитаты («стр. 3», «строки 37–38») или None,
    если цитаты в документе нет. Ищем в одном сегменте, затем в нескольких подряд."""
    q = normalize(quote)
    if len(q) < 4:
        return None
    norm = [normalize(s.text) for s in doc.segments]
    for span in range(1, max_span + 1):
        for i in range(len(norm) - span + 1):
            if not norm[i]:
                continue
            window = " ".join(n for n in norm[i:i + span] if n)
            if q in window:
                first, last = doc.segments[i].number, doc.segments[i + span - 1].number
                if span == 1:
                    return f"{doc.unit_name} {first}"
                prefix = "стр." if doc.kind == "pdf" else "строки"
                return f"{prefix} {first}–{last}"
    return None
