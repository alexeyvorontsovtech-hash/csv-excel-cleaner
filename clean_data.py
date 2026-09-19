#!/usr/bin/env python3
"""
Консольная утилита для очистки, конвертации, разделения и объединения
CSV/Excel файлов.

Примеры запуска:
    python clean_data.py clean data.csv -o clean.csv --all
    python clean_data.py convert data.csv data.xlsx
    python clean_data.py split data.csv --by Категория -o out_dir
    python clean_data.py merge jan.csv feb.csv -o full_year.csv
"""

import argparse
import os
import re
import sys
from datetime import datetime

import pandas as pd

# =====================================================================
# Чтение и запись файлов
# =====================================================================

def read_table(path):
    """Читает CSV или Excel файл в DataFrame. При проблемах — понятная
    ошибка в консоль, без трейсбека."""
    if not os.path.exists(path):
        sys.exit(f"Ошибка: файл не найден: {path}")

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".csv":
            try:
                return pd.read_csv(path, dtype=str)
            except UnicodeDecodeError:
                # Частый случай для выгрузок из 1С — кодировка Windows-1251
                try:
                    return pd.read_csv(path, dtype=str, encoding="cp1251")
                except UnicodeDecodeError:
                    sys.exit(
                        f"Ошибка: не удалось определить кодировку файла {path}. "
                        "Попробуйте пересохранить файл в кодировке UTF-8."
                    )
        elif ext == ".xlsx":
            return pd.read_excel(path, dtype=str)
        else:
            sys.exit(
                f"Ошибка: неподдерживаемый формат файла '{ext}'. "
                "Поддерживаются: .csv, .xlsx"
            )
    except pd.errors.EmptyDataError:
        sys.exit(f"Ошибка: файл пустой или повреждён: {path}")
    except pd.errors.ParserError as e:
        sys.exit(f"Ошибка: не удалось разобрать файл {path}: {e}")


def write_table(df, path):
    """Сохраняет DataFrame в CSV, Excel или JSON — формат определяется
    по расширению файла в path."""
    ext = os.path.splitext(path)[1].lower()
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        if ext == ".csv":
            # utf-8-sig, чтобы кириллица корректно открывалась в Excel
            df.to_csv(path, index=False, encoding="utf-8-sig")
        elif ext == ".xlsx":
            df.to_excel(path, index=False)
        elif ext == ".json":
            df.to_json(path, orient="records", force_ascii=False, indent=2)
        else:
            sys.exit(
                f"Ошибка: неподдерживаемый формат вывода '{ext}'. "
                "Поддерживаются: .csv, .xlsx, .json"
            )
    except PermissionError:
        sys.exit(
            f"Ошибка: нет доступа для записи файла {path} "
            "(возможно, он открыт в другой программе)."
        )


# =====================================================================
# Операции очистки — каждая в своей функции
# =====================================================================

def remove_duplicate_rows(df):
    """Удаляет полностью одинаковые строки. Возвращает новый df и
    количество удалённых строк."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    return df, removed


def remove_empty_rows_and_columns(df):
    """Удаляет строки и столбцы, полностью состоящие из пустых значений."""
    before_rows, before_cols = len(df), len(df.columns)
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    removed_rows = before_rows - len(df)
    removed_cols = before_cols - len(df.columns)
    return df, removed_rows, removed_cols


def trim_whitespace(df):
    """Убирает пробелы в начале/конце значений текстовых полей."""
    df = df.copy()
    changed_count = 0
    for col in df.columns:
        original = df[col]
        stripped = original.apply(lambda v: v.strip() if isinstance(v, str) else v)
        changed_count += int(((stripped != original) & original.notna()).sum())
        df[col] = stripped
    return df, changed_count


DATE_FORMATS = ["%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%y"]
DATE_LIKE_PATTERN = re.compile(r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}$")


def detect_date_columns(df):
    """Автоматически находит столбцы, похожие на даты, по образцу значений."""
    detected = []
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20)
        if sample.empty:
            continue
        matches = sample.apply(lambda v: bool(DATE_LIKE_PATTERN.match(v.strip())))
        if matches.mean() > 0.5:
            detected.append(col)
    return detected


def normalize_dates(df, columns=None):
    """Приводит даты в указанных столбцах к единому формату ДД.ММ.ГГГГ.
    Понимает несколько распространённых форматов на входе."""
    df = df.copy()
    columns = columns if columns is not None else detect_date_columns(df)
    changed_count = 0

    def parse_one(value):
        if not isinstance(value, str) or not value.strip():
            return value, False
        text = value.strip()
        for fmt in DATE_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
                new_value = parsed.strftime("%d.%m.%Y")
                return new_value, new_value != text
            except ValueError:
                continue
        return value, False

    for col in columns:
        results = df[col].apply(parse_one)
        df[col] = results.apply(lambda r: r[0])
        changed_count += int(results.apply(lambda r: r[1]).sum())

    return df, changed_count, columns


NUMBER_LIKE_PATTERN = re.compile(r"^[\d\s.,₽$€]+$")
CURRENCY_PATTERN = re.compile(r"[₽$€]|руб\.?", re.IGNORECASE)


FORMATTING_MARKER_PATTERN = re.compile(r"[,\s₽$€]")


def detect_number_columns(df):
    """Автоматически находит столбцы, похожие на числа с "грязным"
    форматированием (пробелы, запятые, символы валют).

    Столбец берётся в работу только если в нём реально встречается
    признак "грязного" числа — иначе легко спутать, например, с ID
    или уже нормализованной датой (там тоже только цифры и точки)."""
    detected = []
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20)
        if sample.empty:
            continue

        def looks_numeric(v):
            v = v.strip()
            return bool(NUMBER_LIKE_PATTERN.match(v)) and any(ch.isdigit() for ch in v)

        matches = sample.apply(looks_numeric)
        has_formatting_marker = sample.apply(lambda v: bool(FORMATTING_MARKER_PATTERN.search(v))).any()
        if matches.mean() > 0.5 and has_formatting_marker:
            detected.append(col)
    return detected


def normalize_numbers(df, columns=None):
    """Приводит числа к единому виду: убирает пробелы и символы валют,
    заменяет запятую на десятичную точку."""
    df = df.copy()
    columns = columns if columns is not None else detect_number_columns(df)
    changed_count = 0

    def parse_one(value):
        if not isinstance(value, str) or not value.strip():
            return value, False
        original = value.strip()
        text = CURRENCY_PATTERN.sub("", original)
        text = text.replace("\xa0", " ").replace(" ", "").strip()
        if "," in text and "." in text:
            text = text.replace(",", "")  # запятая — разделитель тысяч
        elif "," in text:
            text = text.replace(",", ".")  # запятая — десятичный разделитель
        try:
            number = float(text)
        except ValueError:
            return value, False
        formatted = f"{number:.2f}"
        return formatted, formatted != original

    for col in columns:
        results = df[col].apply(parse_one)
        df[col] = results.apply(lambda r: r[0])
        changed_count += int(results.apply(lambda r: r[1]).sum())

    return df, changed_count, columns


# =====================================================================
# Вспомогательное
# =====================================================================

def parse_column_list(arg_value, df):
    """Разбирает список столбцов из аргумента командной строки
    (через запятую) и проверяет, что они существуют в файле."""
    if not arg_value:
        return None
    columns = [c.strip() for c in arg_value.split(",") if c.strip()]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        sys.exit(
            f"Ошибка: столбцы не найдены в файле: {', '.join(missing)}. "
            f"Доступные столбцы: {', '.join(df.columns)}"
        )
    return columns


def safe_filename_part(value):
    """Превращает значение столбца в безопасный кусок имени файла."""
    text = re.sub(r"[^\w\-]+", "_", str(value)).strip("_")
    return text or "без_значения"


# =====================================================================
# Команды
# =====================================================================

def cmd_clean(args):
    df = read_table(args.input)
    rows_before, cols_before = len(df), len(df.columns)

    any_flag = any([
        args.dedup, args.drop_empty, args.trim,
        args.normalize_dates, args.normalize_numbers, args.all,
    ])
    if not any_flag:
        sys.exit(
            "Ошибка: укажите хотя бы одну операцию очистки "
            "(--dedup, --drop-empty, --trim, --normalize-dates, "
            "--normalize-numbers) или --all"
        )

    report_lines = []

    if args.dedup or args.all:
        df, removed = remove_duplicate_rows(df)
        report_lines.append(f"Удалено дублей строк: {removed}")

    if args.drop_empty or args.all:
        df, removed_rows, removed_cols = remove_empty_rows_and_columns(df)
        report_lines.append(f"Удалено пустых строк: {removed_rows}")
        report_lines.append(f"Удалено пустых столбцов: {removed_cols}")

    if args.trim or args.all:
        df, changed = trim_whitespace(df)
        report_lines.append(f"Обрезаны пробелы в ячейках: {changed}")

    if args.normalize_dates or args.all:
        columns = parse_column_list(args.date_columns, df)
        df, changed, used_columns = normalize_dates(df, columns)
        cols_text = ", ".join(used_columns) if used_columns else "не найдены"
        report_lines.append(f"Нормализовано дат: {changed} (столбцы: {cols_text})")

    if args.normalize_numbers or args.all:
        columns = parse_column_list(args.number_columns, df)
        df, changed, used_columns = normalize_numbers(df, columns)
        cols_text = ", ".join(used_columns) if used_columns else "не найдены"
        report_lines.append(f"Нормализовано чисел: {changed} (столбцы: {cols_text})")

    write_table(df, args.output)

    print("=== Отчёт об очистке файла ===")
    print(f"Входной файл: {args.input}")
    print(f"Строк было: {rows_before} -> стало: {len(df)}")
    print(f"Столбцов было: {cols_before} -> стало: {len(df.columns)}")
    for line in report_lines:
        print(f"- {line}")
    print(f"Результат сохранён в: {args.output}")


def cmd_convert(args):
    df = read_table(args.input)
    write_table(df, args.output)
    print(f"Файл {args.input} сконвертирован в {args.output} ({len(df)} строк)")


def cmd_split(args):
    df = read_table(args.input)
    if args.by not in df.columns:
        sys.exit(
            f"Ошибка: столбец '{args.by}' не найден. "
            f"Доступные столбцы: {', '.join(df.columns)}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    print("=== Отчёт о разделении файла ===")
    print(f"Входной файл: {args.input} ({len(df)} строк)")
    print(f"Разделение по столбцу: {args.by}")

    for value, group in df.groupby(args.by, dropna=False):
        filename = f"{base_name}_{safe_filename_part(value)}.{args.format}"
        out_path = os.path.join(args.output_dir, filename)
        write_table(group, out_path)
        print(f"- {value}: {len(group)} строк -> {out_path}")


def cmd_merge(args):
    dfs = [read_table(path) for path in args.inputs]

    first_columns = list(dfs[0].columns)
    for path, df in zip(args.inputs[1:], dfs[1:]):
        if list(df.columns) != first_columns:
            sys.exit(
                f"Ошибка: структура файла {path} не совпадает с первым файлом.\n"
                f"Ожидались столбцы: {first_columns}\n"
                f"Найдены столбцы: {list(df.columns)}"
            )

    merged = pd.concat(dfs, ignore_index=True)
    write_table(merged, args.output)

    print("=== Отчёт об объединении файлов ===")
    for path, df in zip(args.inputs, dfs):
        print(f"- {path}: {len(df)} строк")
    print(f"Итого строк в объединённом файле: {len(merged)}")
    print(f"Результат сохранён в: {args.output}")


# =====================================================================
# Разбор аргументов командной строки
# =====================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        prog="clean_data.py",
        description="Очистка, конвертация, разделение и объединение CSV/Excel файлов.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- clean ---
    p_clean = subparsers.add_parser("clean", help="Очистить файл от типовых проблем")
    p_clean.add_argument("input", help="Путь к входному файлу (.csv, .xlsx)")
    p_clean.add_argument("-o", "--output", required=True, help="Путь к результату")
    p_clean.add_argument("--dedup", action="store_true", help="Удалить дубли строк")
    p_clean.add_argument("--drop-empty", action="store_true", help="Удалить пустые строки и столбцы")
    p_clean.add_argument("--trim", action="store_true", help="Обрезать пробелы в текстовых полях")
    p_clean.add_argument("--normalize-dates", action="store_true", help="Привести даты к единому формату")
    p_clean.add_argument("--date-columns", help="Столбцы с датами через запятую (по умолчанию — автоопределение)")
    p_clean.add_argument("--normalize-numbers", action="store_true", help="Привести числа к единому формату")
    p_clean.add_argument("--number-columns", help="Столбцы с числами через запятую (по умолчанию — автоопределение)")
    p_clean.add_argument("--all", action="store_true", help="Включить все операции очистки")
    p_clean.set_defaults(func=cmd_clean)

    # --- convert ---
    p_convert = subparsers.add_parser("convert", help="Конвертировать файл между CSV, Excel и JSON")
    p_convert.add_argument("input", help="Путь к входному файлу")
    p_convert.add_argument("output", help="Путь к результату (формат определяется по расширению)")
    p_convert.set_defaults(func=cmd_convert)

    # --- split ---
    p_split = subparsers.add_parser("split", help="Разделить файл на несколько по значению столбца")
    p_split.add_argument("input", help="Путь к входному файлу")
    p_split.add_argument("--by", required=True, help="Название столбца для разделения")
    p_split.add_argument("-o", "--output-dir", default="split_output", help="Папка для результатов")
    p_split.add_argument("--format", choices=["csv", "xlsx", "json"], default="csv", help="Формат файлов на выходе")
    p_split.set_defaults(func=cmd_split)

    # --- merge ---
    p_merge = subparsers.add_parser("merge", help="Объединить несколько файлов с одинаковой структурой")
    p_merge.add_argument("inputs", nargs="+", help="Пути к входным файлам")
    p_merge.add_argument("-o", "--output", required=True, help="Путь к результату")
    p_merge.set_defaults(func=cmd_merge)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
