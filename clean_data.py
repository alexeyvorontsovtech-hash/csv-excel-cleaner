#!/usr/bin/env python3
"""
Консольная утилита для очистки, конвертации, разделения и объединения
CSV/Excel файлов.

Примеры запуска:
    python clean_data.py clean data.csv -o clean.csv --all
    python clean_data.py convert data.csv data.xlsx
    python clean_data.py split data.csv --by Категория -o out_dir
    python clean_data.py merge jan.csv feb.csv -o full_year.csv
    python clean_data.py clean data.csv -o clean.csv --all --sep ";" --out-sep ","
"""

import argparse
import csv
import decimal
import os
import re
import sys
from datetime import datetime

import pandas as pd

# =====================================================================
# Чтение и запись файлов
# =====================================================================

CSV_SEPARATOR_CANDIDATES = [";", ",", "\t"]


def detect_csv_separator(sample_text):
    """Определяет разделитель CSV по образцу текста: для каждого
    кандидата (';', ',', таб) разбирает первые строки с учётом кавычек
    и смотрит, сколько полей получается на строку. Выбирается кандидат,
    дающий одинаковое (и большее одного) число полей на всех строках —
    это надёжнее, чем просто частота символа, которую могут исказить
    запятые внутри текстовых значений. Запятая — разделитель по
    умолчанию, если ни один из кандидатов не подошёл."""
    first_lines = [ln for ln in sample_text.splitlines()[:5] if ln != ""]
    if not first_lines:
        return ","

    best_sep = None
    best_score = None
    for sep in CSV_SEPARATOR_CANDIDATES:
        try:
            field_counts = [len(row) for row in csv.reader(first_lines, delimiter=sep)]
        except csv.Error:
            continue
        if not field_counts or max(field_counts) <= 1:
            continue
        consistent = len(set(field_counts)) == 1
        score = (consistent, field_counts[0])
        if best_score is None or score > best_score:
            best_score = score
            best_sep = sep
    return best_sep if best_sep is not None else ","


def _read_csv_sample(path, encoding, size=4096):
    with open(path, "r", encoding=encoding, newline="") as f:
        return f.read(size)


def _validate_csv_field_counts(path, encoding, sep):
    """Проверяет, что каждая строка содержит столько же полей, сколько в
    заголовке. Иначе pandas может молча сдвинуть лишнее поле в индекс
    (или потерять его) вместо явной ошибки."""
    with open(path, "r", encoding=encoding, newline="") as f:
        rows = [row for row in csv.reader(f, delimiter=sep) if row != []]
    if not rows:
        return
    header_len = len(rows[0])
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != header_len:
            sys.exit(
                f"Ошибка: файл {path}, строка {line_number}: {len(row)} "
                f"полей вместо {header_len} (как в заголовке). "
                "Проверьте разделитель (--sep) и кавычки в значениях."
            )


BAD_ENCODING_RATIO_THRESHOLD = 0.01


def _looks_like_wrong_encoding(text):
    """Оценивает, похож ли декодированный текст на результат чтения в
    неверной кодировке: считает долю replacement-символов (U+FFFD) и
    управляющих символов (кроме обычных \\n, \\r, \\t)."""
    if not text:
        return False
    bad = sum(
        1 for ch in text
        if ch == "�" or (ord(ch) < 32 and ch not in "\n\r\t")
    )
    return (bad / len(text)) > BAD_ENCODING_RATIO_THRESHOLD


def read_table(path, sep=None):
    """Читает CSV или Excel файл в DataFrame. При проблемах — понятная
    ошибка в консоль, без трейсбека.

    Для CSV разделитель по умолчанию определяется автоматически
    (';', ',', таб) по образцу файла; можно задать явно через sep."""
    if not os.path.exists(path):
        sys.exit(f"Ошибка: файл не найден: {path}")

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".csv":
            for encoding in ("utf-8", "cp1251"):
                try:
                    sample = _read_csv_sample(path, encoding)
                    if _looks_like_wrong_encoding(sample):
                        continue
                    used_sep = (
                        sep if sep is not None
                        else detect_csv_separator(sample)
                    )
                    _validate_csv_field_counts(path, encoding, used_sep)
                    return pd.read_csv(
                        path, dtype=str, sep=used_sep, encoding=encoding,
                        keep_default_na=False, na_values=[""],
                    )
                except UnicodeDecodeError:
                    continue
            sys.exit(
                f"Ошибка: не удалось определить кодировку файла {path}. "
                "Попробуйте пересохранить файл в кодировке UTF-8."
            )
        elif ext == ".xlsx":
            try:
                return pd.read_excel(
                    path, dtype=str, keep_default_na=False, na_values=[""],
                )
            except Exception as e:
                sys.exit(f"Ошибка: не удалось прочитать xlsx-файл {path}: {e}")
        else:
            sys.exit(
                f"Ошибка: неподдерживаемый формат файла '{ext}'. "
                "Поддерживаются: .csv, .xlsx"
            )
    except pd.errors.EmptyDataError:
        sys.exit(f"Ошибка: файл пустой или повреждён: {path}")
    except pd.errors.ParserError as e:
        sys.exit(f"Ошибка: не удалось разобрать файл {path}: {e}")


FORMULA_TRIGGER_CHARS = ("=", "+", "-", "@")


def escape_formula_like_text(df):
    """Экранирует текстовые значения, похожие на формулы Excel (начинаются
    с =, +, - или @), префиксом апострофа — иначе Excel может исполнить их
    как формулу вместо того, чтобы показать как исходный текст."""
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(
            lambda v: f"'{v}" if isinstance(v, str) and v.startswith(FORMULA_TRIGGER_CHARS) else v
        )
    return df


def write_table(df, path, sep=","):
    """Сохраняет DataFrame в CSV, Excel или JSON — формат определяется
    по расширению файла в path. sep задаёт разделитель для CSV."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv" and len(sep) != 1:
        sys.exit(
            f"Ошибка: разделитель для вывода (--out-sep) должен быть "
            f"ровно одним символом, получено: '{sep}'."
        )

    out_dir = os.path.dirname(path)
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            sys.exit(f"Ошибка: не удалось создать каталог {out_dir}: {e}")

    try:
        if ext == ".csv":
            # utf-8-sig, чтобы кириллица корректно открывалась в Excel
            df.to_csv(path, index=False, encoding="utf-8-sig", sep=sep)
        elif ext == ".xlsx":
            escape_formula_like_text(df).to_excel(path, index=False)
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


def _strip_value(v):
    """Обрезает пробелы у строки; строка, ставшая пустой после
    обрезки, считается пропуском (NaN), а не пустой строкой — иначе
    dropna() не удаляет такие "пустые" ячейки."""
    if not isinstance(v, str):
        return v
    stripped = v.strip()
    return stripped if stripped else float("nan")


def trim_whitespace(df):
    """Убирает пробелы в начале/конце значений текстовых полей."""
    df = df.copy()
    changed_count = 0
    for col in df.columns:
        original = df[col]
        stripped = original.apply(_strip_value)
        changed_count += int(((stripped != original) & original.notna()).sum())
        df[col] = stripped
    return df, changed_count


DATE_FORMATS = [
    ("%d.%m.%Y", "%d.%m.%Y"),
    ("%Y-%m-%d", "%d.%m.%Y"),
    ("%d/%m/%Y", "%d.%m.%Y"),
    ("%d-%m-%Y", "%d.%m.%Y"),
    ("%d.%m.%y", "%d.%m.%Y"),
    ("%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S"),
    ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M:%S"),
]
DATE_LIKE_PATTERN = re.compile(
    r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}( \d{1,2}:\d{1,2}:\d{1,2})?$"
)


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
    """Приводит даты в указанных столбцах к единому формату ДД.ММ.ГГГГ
    (с временем — ДД.ММ.ГГГГ ЧЧ:ММ:СС, если время было в исходном
    значении). Понимает несколько распространённых форматов на входе.

    Значения, похожие по форме на дату, но не существующие (например,
    31.02.2024), не изменяются и считаются в invalid_count —
    "некорректная дата". Значения, вообще не похожие на дату ни в одном
    из поддерживаемых форматов, считаются отдельно в unsupported_count —
    "неподдерживаемый формат"."""
    df = df.copy()
    columns = columns if columns is not None else detect_date_columns(df)
    changed_count = 0
    invalid_count = 0
    unsupported_count = 0

    def parse_one(value):
        if not isinstance(value, str) or not value.strip():
            return value, False, False, False
        text = value.strip()
        for in_fmt, out_fmt in DATE_FORMATS:
            try:
                parsed = datetime.strptime(text, in_fmt)
            except ValueError:
                continue
            new_value = parsed.strftime(out_fmt)
            return new_value, new_value != text, False, False
        looks_like_date = bool(DATE_LIKE_PATTERN.match(text))
        return value, False, looks_like_date, not looks_like_date

    for col in columns:
        results = df[col].apply(parse_one)
        df[col] = results.apply(lambda r: r[0])
        changed_count += int(results.apply(lambda r: r[1]).sum())
        invalid_count += int(results.apply(lambda r: r[2]).sum())
        unsupported_count += int(results.apply(lambda r: r[3]).sum())

    return df, changed_count, invalid_count, unsupported_count, columns


NUMBER_LIKE_PATTERN = re.compile(r"^-?[\d\s.,]+$")
CURRENCY_PATTERN = re.compile(r"[₽$€]|руб\.?", re.IGNORECASE)
FORMATTING_MARKER_PATTERN = re.compile(r"[,\s₽$€-]")
SCIENTIFIC_NOTATION_PATTERN = re.compile(r"^-?\d+(\.\d+)?[eE][+-]?\d+$")


def detect_number_columns(df, exclude=None):
    """Автоматически находит столбцы, похожие на числа с "грязным"
    форматированием (пробелы, запятые, символы валют, минус).

    Столбец берётся в работу только если в нём реально встречается
    признак "грязного" числа — иначе легко спутать, например, с ID
    или уже нормализованной датой (там тоже только цифры и точки).

    exclude — столбцы, которые не рассматриваются (например, уже
    распознанные как даты — невалидная дата вида "31.02.2024" содержит
    дефис и без исключения может провоцировать ложное срабатывание)."""
    exclude = exclude or []
    detected = []
    for col in df.columns:
        if col in exclude:
            continue
        sample = df[col].dropna().astype(str).head(20)
        if sample.empty:
            continue

        def looks_numeric(v):
            v = v.strip()
            stripped = CURRENCY_PATTERN.sub("", v).strip()
            return bool(NUMBER_LIKE_PATTERN.match(stripped)) and any(
                ch.isdigit() for ch in stripped
            )

        matches = sample.apply(looks_numeric)
        has_formatting_marker = sample.apply(
            lambda v: bool(FORMATTING_MARKER_PATTERN.search(v))
            or bool(CURRENCY_PATTERN.search(v))
        ).any()
        if matches.mean() > 0.5 and has_formatting_marker:
            detected.append(col)
    return detected


def normalize_numbers(df, columns=None, exclude=None):
    """Приводит числа к единому виду: убирает пробелы и символы валют,
    определяет десятичный разделитель и приводит его к точке. Не
    округляет — сохраняет исходную точность.

    Если в числе встречаются и запятая, и точка — десятичным считается
    тот символ, что находится правее (например, "1.234,56" -> 1234.56).

    Если в числе только запятая и ровно три цифры после неё, а перед
    запятой не только "0" — это неоднозначный случай (может быть и
    разделитель тысяч, и десятичный разделитель): значение не
    изменяется и учитывается в unrecognized_count — "некорректное
    число". Значения, вообще не похожие на число (не удалось разобрать
    после снятия форматирования), считаются отдельно в
    unsupported_count — "неподдерживаемый формат"."""
    df = df.copy()
    columns = columns if columns is not None else detect_number_columns(df, exclude=exclude)
    changed_count = 0
    unrecognized_count = 0
    unsupported_count = 0

    def parse_one(value):
        if not isinstance(value, str) or not value.strip():
            return value, False, False, False
        original = value.strip()
        text = CURRENCY_PATTERN.sub("", original)
        text = text.replace("\xa0", " ").replace(" ", "").strip()
        if not text:
            return value, False, False, True

        if SCIENTIFIC_NOTATION_PATTERN.match(text):
            try:
                number = decimal.Decimal(text)
            except decimal.InvalidOperation:
                return value, False, False, True
            formatted = format(number, "f")
            return formatted, formatted != original, False, False

        has_comma = "," in text
        has_dot = "." in text

        if has_comma and has_dot:
            if text.rindex(",") > text.rindex("."):
                decimal_sep, thousands_sep = ",", "."
            else:
                decimal_sep, thousands_sep = ".", ","
            text = text.replace(thousands_sep, "")
            if decimal_sep != ".":
                text = text.replace(decimal_sep, ".")
        elif has_comma:
            if text.count(",") > 1:
                text = text.replace(",", "")
            else:
                integer_part, _, fraction_part = text.partition(",")
                if len(fraction_part) == 3 and integer_part.lstrip("-") not in ("", "0"):
                    return value, False, True, False
                text = text.replace(",", ".")
        elif has_dot:
            if text.count(".") > 1:
                text = text.replace(".", "")

        try:
            number = decimal.Decimal(text)
        except decimal.InvalidOperation:
            return value, False, False, True

        decimals = len(text.split(".")[-1]) if "." in text else 0
        formatted = f"{number:.{decimals}f}"
        return formatted, formatted != original, False, False

    for col in columns:
        results = df[col].apply(parse_one)
        df[col] = results.apply(lambda r: r[0])
        changed_count += int(results.apply(lambda r: r[1]).sum())
        unrecognized_count += int(results.apply(lambda r: r[2]).sum())
        unsupported_count += int(results.apply(lambda r: r[3]).sum())

    return df, changed_count, unrecognized_count, unsupported_count, columns


NUMERIC_STRING_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")


def coerce_numeric_columns(df, columns):
    """Приводит нормализованные числовые столбцы к настоящему числовому
    типу — нужно перед записью в xlsx/json, чтобы там были числа, а не
    текст. Значения, которые не удалось распознать как число, остаются
    как есть."""
    df = df.copy()

    def to_number(value):
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not NUMERIC_STRING_PATTERN.match(text):
            return value
        return float(text) if "." in text else int(text)

    for col in columns:
        # dtype=object явно — иначе pandas при сборке Series из смеси
        # int и float автоматически повышает всё до float64, и "2500"
        # превращается в 2500.0 вместо целого числа.
        df[col] = pd.Series(
            [to_number(v) for v in df[col]], index=df.index, dtype=object
        )
    return df


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
    """Превращает значение столбца в безопасный кусок имени файла.
    Пустое значение (NaN) превращается в "без_значения"."""
    if pd.isna(value):
        return "без_значения"
    text = re.sub(r"[^\w\-]+", "_", str(value)).strip("_")
    return text or "без_значения"


# =====================================================================
# Команды
# =====================================================================

def cmd_clean(args):
    df = read_table(args.input, sep=args.sep)
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
    number_columns_used = []
    date_columns_used = []

    # Порядок важен: сначала приводим значения к единому виду
    # (пробелы, пустые строки, даты, числа) и только потом ищем
    # дубли — так дублями считаются и строки, различавшиеся только
    # форматированием.
    if args.trim or args.all:
        df, changed = trim_whitespace(df)
        report_lines.append(f"Обрезаны пробелы в ячейках: {changed}")

    if args.drop_empty or args.all:
        df, removed_rows, removed_cols = remove_empty_rows_and_columns(df)
        report_lines.append(f"Удалено пустых строк: {removed_rows}")
        report_lines.append(f"Удалено пустых столбцов: {removed_cols}")

    if args.normalize_dates or args.all:
        columns = parse_column_list(args.date_columns, df)
        df, changed, invalid, unsupported, used_columns = normalize_dates(df, columns)
        date_columns_used = used_columns
        cols_text = ", ".join(used_columns) if used_columns else "не найдены"
        report_lines.append(
            f"Нормализовано дат: {changed} (столбцы: {cols_text}), "
            f"некорректных дат: {invalid}, неподдерживаемый формат: {unsupported}"
        )

    if args.normalize_numbers or args.all:
        columns = parse_column_list(args.number_columns, df)
        df, changed, unrecognized, unsupported, used_columns = normalize_numbers(
            df, columns, exclude=date_columns_used
        )
        number_columns_used = used_columns
        cols_text = ", ".join(used_columns) if used_columns else "не найдены"
        report_lines.append(
            f"Нормализовано чисел: {changed} (столбцы: {cols_text}), "
            f"некорректных чисел: {unrecognized}, неподдерживаемый формат: {unsupported}"
        )

    if args.dedup or args.all:
        df, removed = remove_duplicate_rows(df)
        report_lines.append(f"Удалено дублей строк: {removed}")

    output_ext = os.path.splitext(args.output)[1].lower()
    if number_columns_used and output_ext in (".xlsx", ".json"):
        df = coerce_numeric_columns(df, number_columns_used)

    write_table(df, args.output, sep=args.out_sep)

    print("=== Отчёт об очистке файла ===")
    print(f"Входной файл: {args.input}")
    print(f"Строк было: {rows_before} -> стало: {len(df)}")
    print(f"Столбцов было: {cols_before} -> стало: {len(df.columns)}")
    for line in report_lines:
        print(f"- {line}")
    print(f"Результат сохранён в: {args.output}")


def cmd_convert(args):
    df = read_table(args.input, sep=args.sep)
    write_table(df, args.output, sep=args.out_sep)
    print(f"Файл {args.input} сконвертирован в {args.output} ({len(df)} строк)")


def cmd_split(args):
    df = read_table(args.input, sep=args.sep)
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

    used_final_stems = set()
    for value, group in df.groupby(args.by, dropna=False):
        stem = f"{base_name}_{safe_filename_part(value)}"
        final_stem = stem
        suffix = 2
        while final_stem in used_final_stems:
            final_stem = f"{stem}_{suffix}"
            suffix += 1
        used_final_stems.add(final_stem)
        out_path = os.path.join(args.output_dir, f"{final_stem}.{args.format}")
        write_table(group, out_path, sep=args.out_sep)
        print(f"- {value}: {len(group)} строк -> {out_path}")


def cmd_merge(args):
    dfs = [read_table(path, sep=args.sep) for path in args.inputs]

    first_columns = list(dfs[0].columns)
    for path, df in zip(args.inputs[1:], dfs[1:]):
        if list(df.columns) != first_columns:
            sys.exit(
                f"Ошибка: структура файла {path} не совпадает с первым файлом.\n"
                f"Ожидались столбцы: {first_columns}\n"
                f"Найдены столбцы: {list(df.columns)}"
            )

    merged = pd.concat(dfs, ignore_index=True)
    write_table(merged, args.output, sep=args.out_sep)

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

    sep_help = "Разделитель CSV на входе (по умолчанию — автоопределение: ';', ',', таб)"
    out_sep_help = "Разделитель CSV на выходе (по умолчанию ',')"

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
    p_clean.add_argument("--sep", help=sep_help)
    p_clean.add_argument("--out-sep", default=",", help=out_sep_help)
    p_clean.set_defaults(func=cmd_clean)

    # --- convert ---
    p_convert = subparsers.add_parser("convert", help="Конвертировать файл между CSV, Excel и JSON")
    p_convert.add_argument("input", help="Путь к входному файлу")
    p_convert.add_argument("output", help="Путь к результату (формат определяется по расширению)")
    p_convert.add_argument("--sep", help=sep_help)
    p_convert.add_argument("--out-sep", default=",", help=out_sep_help)
    p_convert.set_defaults(func=cmd_convert)

    # --- split ---
    p_split = subparsers.add_parser("split", help="Разделить файл на несколько по значению столбца")
    p_split.add_argument("input", help="Путь к входному файлу")
    p_split.add_argument("--by", required=True, help="Название столбца для разделения")
    p_split.add_argument("-o", "--output-dir", default="split_output", help="Папка для результатов")
    p_split.add_argument("--format", choices=["csv", "xlsx", "json"], default="csv", help="Формат файлов на выходе")
    p_split.add_argument("--sep", help=sep_help)
    p_split.add_argument("--out-sep", default=",", help=out_sep_help)
    p_split.set_defaults(func=cmd_split)

    # --- merge ---
    p_merge = subparsers.add_parser("merge", help="Объединить несколько файлов с одинаковой структурой")
    p_merge.add_argument("inputs", nargs="+", help="Пути к входным файлам")
    p_merge.add_argument("-o", "--output", required=True, help="Путь к результату")
    p_merge.add_argument("--sep", help=sep_help)
    p_merge.add_argument("--out-sep", default=",", help=out_sep_help)
    p_merge.set_defaults(func=cmd_merge)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
