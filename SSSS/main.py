import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.agent import run_validation
from src.document import DocumentError
from src.llm import LLMError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ИИ-агент статической проверки документов требований (ISO/IEC/IEEE 29148:2018 & IEEE 830-1998)."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Путь к проверяемому документу требований (.pdf, .md, .txt)",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Путь для сохранения итогового отчёта в формате Markdown (.md)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Название модели Gemini (по умолчанию берётся из GEMINI_MODEL или gemini-3.8-flash)",
    )
    return parser.parse_args()


def main() -> None:
    # Кириллица в консоли Windows
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    load_dotenv()
    args = parse_args()

    if args.output_path.suffix.lower() != ".md":
        print("Предупреждение: отчёт пишется в формате Markdown, рекомендуется расширение .md")
    if args.input_path.resolve() == args.output_path.resolve():
        print("Ошибка: путь отчёта совпадает с путём входного документа.", file=sys.stderr)
        sys.exit(2)

    try:
        run_validation(
            input_path=args.input_path,
            output_path=args.output_path,
            model_name=args.model,
        )
    except (DocumentError, LLMError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Прервано пользователем.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Ошибка при выполнении проверки: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
