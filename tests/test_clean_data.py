"""
Тесты для clean_data.py.

Функции очистки тестируются напрямую через DataFrame, без запуска CLI.
Отдельно — несколько CLI-тестов через subprocess для проверки кодов
возврата и текста ошибок.
"""

import json
import subprocess
import sys
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import clean_data  # noqa: E402

CLEAN_DATA_SCRIPT = PROJECT_ROOT / "clean_data.py"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(CLEAN_DATA_SCRIPT), *args],
        capture_output=True,
        text=True,
    )


# =====================================================================
# remove_duplicate_rows
# =====================================================================

def test_remove_duplicate_rows_removes_and_counts():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    result, removed = clean_data.remove_duplicate_rows(df)
    assert removed == 1
    assert len(result) == 2


def test_remove_duplicate_rows_no_duplicates():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result, removed = clean_data.remove_duplicate_rows(df)
    assert removed == 0
    assert len(result) == 3


# =====================================================================
# remove_empty_rows_and_columns
# =====================================================================

def test_remove_empty_rows_and_columns():
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": ["x", None, "z"],
        "c": [None, None, None],
    })
    result, removed_rows, removed_cols = clean_data.remove_empty_rows_and_columns(df)
    assert removed_rows == 1
    assert removed_cols == 1
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 2


def test_remove_empty_rows_and_columns_nothing_to_remove():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result, removed_rows, removed_cols = clean_data.remove_empty_rows_and_columns(df)
    assert removed_rows == 0
    assert removed_cols == 0
    assert len(result) == 2
    assert list(result.columns) == ["a", "b"]


# =====================================================================
# trim_whitespace
# =====================================================================

def test_trim_whitespace_strips_and_counts():
    df = pd.DataFrame({"name": ["  Иванов И.И. ", "Петров"]})
    result, changed = clean_data.trim_whitespace(df)
    assert changed == 1
    assert repr(result["name"].iloc[0]) == repr("Иванов И.И.")
    assert result["name"].iloc[1] == "Петров"


def test_trim_whitespace_blank_string_becomes_nan():
    """Строка из одних пробелов после обрезки становится "" — но dropna()
    не считает "" пустым значением, поэтому такая "пустая" ячейка не
    удаляется. Обрезка пустой после trim строки должна давать NaN."""
    df = pd.DataFrame({"name": ["   ", "Иванов"]})
    result, changed = clean_data.trim_whitespace(df)
    assert pd.isna(result["name"].iloc[0])
    assert result["name"].iloc[1] == "Иванов"
    assert changed == 1


def test_trim_whitespace_with_pandas_string_dtype():
    """Регрессионный тест: раньше trim_whitespace() пропускал столбцы,
    у которых dtype не равен object (например, pandas StringDtype или
    новый строковый dtype в pandas 3.0), и молча ничего не обрезал."""
    df = pd.DataFrame({"name": pd.array([" Иванов ", "Петров"], dtype="string")})
    result, changed = clean_data.trim_whitespace(df)
    assert changed == 1
    assert result["name"].iloc[0] == "Иванов"
    assert result["name"].iloc[1] == "Петров"


# =====================================================================
# normalize_dates
# =====================================================================

@pytest.mark.parametrize("raw,expected", [
    ("05.03.2024", "05.03.2024"),
    ("2024-03-06", "06.03.2024"),
    ("07/03/2024", "07.03.2024"),
])
def test_normalize_dates_known_formats(raw, expected):
    df = pd.DataFrame({"date": [raw]})
    result, changed, invalid, unsupported, columns = clean_data.normalize_dates(df, columns=["date"])
    assert result["date"].iloc[0] == expected
    assert columns == ["date"]
    assert invalid == 0


def test_normalize_dates_non_date_value_unchanged():
    df = pd.DataFrame({"date": ["не дата"]})
    result, changed, invalid, unsupported, columns = clean_data.normalize_dates(df, columns=["date"])
    assert result["date"].iloc[0] == "не дата"
    assert changed == 0
    assert invalid == 0


def test_normalize_dates_datetime_with_time_supported():
    df = pd.DataFrame({"date": ["2024-03-06 10:15:00"]})
    result, changed, invalid, unsupported, columns = clean_data.normalize_dates(df, columns=["date"])
    assert result["date"].iloc[0] == "06.03.2024 10:15:00"
    assert changed == 1
    assert invalid == 0


def test_normalize_dates_invalid_date_unchanged_but_counted():
    df = pd.DataFrame({"date": ["31.02.2024"]})
    result, changed, invalid, unsupported, columns = clean_data.normalize_dates(df, columns=["date"])
    assert result["date"].iloc[0] == "31.02.2024"
    assert changed == 0
    assert invalid == 1
    assert unsupported == 0


def test_normalize_dates_unsupported_format_counted_separately_from_invalid():
    """"31.02.2024" похоже на дату по форме, но такого числа не бывает —
    это "некорректная дата". "не дата" вообще не похоже на дату — это
    "неподдерживаемый формат". Отчёт должен различать эти два случая."""
    df = pd.DataFrame({"date": ["не дата", "31.02.2024"]})
    result, changed, invalid, unsupported, columns = clean_data.normalize_dates(df, columns=["date"])
    assert invalid == 1
    assert unsupported == 1


# =====================================================================
# normalize_numbers
# =====================================================================

@pytest.mark.parametrize("raw,expected", [
    ("1 200,50", "1200.50"),
    ("2 500,00 ₽", "2500.00"),
    ("980.00", "980.00"),
])
def test_normalize_numbers_known_formats(raw, expected):
    df = pd.DataFrame({"amount": [raw]})
    result, changed, unrecognized, unsupported, columns = clean_data.normalize_numbers(df, columns=["amount"])
    assert result["amount"].iloc[0] == expected


def test_normalize_numbers_non_numeric_value_unchanged():
    df = pd.DataFrame({"amount": ["не число"]})
    result, changed, unrecognized, unsupported, columns = clean_data.normalize_numbers(df, columns=["amount"])
    assert result["amount"].iloc[0] == "не число"
    assert changed == 0
    assert unrecognized == 0
    assert unsupported == 1


def test_normalize_numbers_unsupported_format_counted_separately_from_invalid():
    """"1,234" похоже на число, но неоднозначно (тысячи или десятичная
    часть?) — это "некорректное число". "не число" вообще не похоже на
    число — это "неподдерживаемый формат". Отчёт должен различать их."""
    df = pd.DataFrame({"amount": ["не число", "1,234"]})
    result, changed, unrecognized, unsupported, columns = clean_data.normalize_numbers(df, columns=["amount"])
    assert unrecognized == 1
    assert unsupported == 1


@pytest.mark.parametrize("raw,expected", [
    ("1.234,56", "1234.56"),
    ("0,125", "0.125"),
    ("-1 200,50", "-1200.50"),
    ("2 500 руб.", "2500"),
])
def test_normalize_numbers_new_cases(raw, expected):
    df = pd.DataFrame({"amount": [raw]})
    result, changed, unrecognized, unsupported, columns = clean_data.normalize_numbers(df, columns=["amount"])
    assert result["amount"].iloc[0] == expected
    assert unrecognized == 0


def test_normalize_numbers_does_not_round_to_two_decimals():
    df = pd.DataFrame({"amount": ["0,125"]})
    result, changed, unrecognized, unsupported, columns = clean_data.normalize_numbers(df, columns=["amount"])
    assert result["amount"].iloc[0] == "0.125"


def test_normalize_numbers_ambiguous_comma_is_not_changed():
    """Ровно 3 цифры после единственной запятой при непустой ненулевой
    целой части — неоднозначный случай (тысячи или десятичная часть?).
    Значение не меняется, но учитывается в unrecognized_count."""
    df = pd.DataFrame({"amount": ["1,234"]})
    result, changed, unrecognized, unsupported, columns = clean_data.normalize_numbers(df, columns=["amount"])
    assert result["amount"].iloc[0] == "1,234"
    assert changed == 0
    assert unrecognized == 1
    assert unsupported == 0


def test_normalize_numbers_preserves_large_integer_precision():
    """float не может точно представить целые числа больше 2**53;
    normalize_numbers должен использовать decimal.Decimal, чтобы не
    склеивать соседние большие числа в одно значение."""
    df = pd.DataFrame({"amount": ["9 007 199 254 740 993", "9007199254740992"]})
    result, changed, unrecognized, unsupported, columns = clean_data.normalize_numbers(df, columns=["amount"])
    assert result["amount"].iloc[0] == "9007199254740993"
    assert result["amount"].iloc[1] == "9007199254740992"
    assert result["amount"].iloc[0] != result["amount"].iloc[1]


def test_normalize_numbers_leading_zero_comma_is_not_ambiguous():
    """"0,125" не может быть тысячами (0125 не бывает), поэтому это
    однозначно десятичная запятая, а не неоднозначный случай."""
    df = pd.DataFrame({"amount": ["0,125"]})
    result, changed, unrecognized, unsupported, columns = clean_data.normalize_numbers(df, columns=["amount"])
    assert unrecognized == 0
    assert changed == 1


# =====================================================================
# coerce_numeric_columns
# =====================================================================

def test_coerce_numeric_columns_converts_strings_to_numbers():
    df = pd.DataFrame({"amount": ["1200.50", "2500"]})
    result = clean_data.coerce_numeric_columns(df, ["amount"])
    assert result["amount"].iloc[0] == pytest.approx(1200.50)
    assert isinstance(result["amount"].iloc[0], float)
    assert result["amount"].iloc[1] == 2500
    assert isinstance(result["amount"].iloc[1], int)


def test_coerce_numeric_columns_keeps_unrecognized_values_untouched():
    df = pd.DataFrame({"amount": ["не число"]})
    result = clean_data.coerce_numeric_columns(df, ["amount"])
    assert result["amount"].iloc[0] == "не число"


# =====================================================================
# safe_filename_part
# =====================================================================

def test_safe_filename_part_nan_becomes_without_value():
    assert clean_data.safe_filename_part(float("nan")) == "без_значения"


def test_safe_filename_part_sanitizes_special_characters():
    assert clean_data.safe_filename_part("A.B") == "A_B"


# =====================================================================
# --all: столбцы дат не должны попадать в автообнаружение чисел
# =====================================================================

def test_cli_clean_all_excludes_date_columns_from_number_autodetection(tmp_path):
    """Невалидная дата (31.02.2024) в столбце дат содержит дефис, который
    без исключения провоцирует автообнаружение того же столбца как
    числового и портит уже нормализованные даты (05.03.2024 -> 5032024)."""
    input_file = tmp_path / "in.csv"
    input_file.write_text(
        "id,date\n"
        "1,2024-03-05\n"
        "2,2024-03-06\n"
        "3,2024-02-31\n"
        "4,2024-03-07\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.csv"
    result = run_cli("clean", str(input_file), "-o", str(output), "--all")
    assert result.returncode == 0
    df = pd.read_csv(output, dtype=str)
    assert list(df["date"]) == [
        "05.03.2024", "06.03.2024", "2024-02-31", "07.03.2024",
    ]


# =====================================================================
# CSV separator autodetection
# =====================================================================

@pytest.mark.parametrize("text,expected_sep", [
    ("a;b;c\n1;2;3\n", ";"),
    ("a,b,c\n1,2,3\n", ","),
    ("a\tb\tc\n1\t2\t3\n", "\t"),
])
def test_detect_csv_separator(text, expected_sep):
    assert clean_data.detect_csv_separator(text) == expected_sep


def test_detect_csv_separator_uses_field_count_not_char_frequency():
    """";" встречается 3 раза (1 раз на строку), а "," — 4 раза, потому
    что комментарий содержит запятые. Частотный подсчёт символов выбрал
    бы ",", хотя реальный разделитель — ";" (по числу полей на строку)."""
    text = (
        "id;comment\n"
        "1;товар,услуга,доставка\n"
        "2;товар,услуга,доставка\n"
    )
    assert clean_data.detect_csv_separator(text) == ";"


def test_read_table_keeps_full_structure_with_commas_inside_values(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text(
        "id;comment\n"
        "1;товар,услуга,доставка\n"
        "2;товар,услуга,доставка\n",
        encoding="utf-8",
    )
    df = clean_data.read_table(str(path))
    assert list(df.columns) == ["id", "comment"]
    assert list(df["id"]) == ["1", "2"]
    assert df["comment"].iloc[0] == "товар,услуга,доставка"


def test_cli_convert_rejects_row_with_more_fields_than_header(tmp_path):
    """Раньше pandas молча сдвигал лишнее поле в индекс: id=10, amount=50,
    а исходный id "1" и последнее значение "50" терялись без предупреждения."""
    input_file = tmp_path / "badrows.csv"
    input_file.write_text("id,amount\n1,10,50\n", encoding="utf-8")
    result = run_cli("convert", str(input_file), str(tmp_path / "out.json"), "--sep", ",")
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in combined
    assert "Ошибка" in combined


def test_read_table_autodetects_semicolon(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
    df = clean_data.read_table(str(path))
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_read_table_autodetects_tab(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a\tb\n1\t2\n", encoding="utf-8")
    df = clean_data.read_table(str(path))
    assert list(df.columns) == ["a", "b"]


@pytest.mark.parametrize("marker", ["NA", "NULL", "N/A", "nan"])
def test_read_table_keeps_na_like_strings_as_text(tmp_path, marker):
    """"NA", "NULL", "N/A", "nan" — валидные текстовые значения
    (например, артикул или код), а не признак пропуска."""
    path = tmp_path / "data.csv"
    path.write_text(f"code,name\n{marker},x\n", encoding="utf-8")
    df = clean_data.read_table(str(path))
    assert df["code"].iloc[0] == marker
    assert not pd.isna(df["code"].iloc[0])


def test_read_table_real_empty_cell_is_still_missing(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("code,name\n,x\n", encoding="utf-8")
    df = clean_data.read_table(str(path))
    assert pd.isna(df["code"].iloc[0])


def test_read_table_sep_override(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a|b\n1|2\n", encoding="utf-8")
    df = clean_data.read_table(str(path), sep="|")
    assert list(df.columns) == ["a", "b"]


def test_cli_clean_out_sep_writes_custom_separator(tmp_path):
    input_file = tmp_path / "in.csv"
    input_file.write_text("a;b\n1;2\n", encoding="utf-8")
    output = tmp_path / "out.csv"
    result = run_cli(
        "clean", str(input_file), "-o", str(output), "--dedup", "--out-sep", ";",
    )
    assert result.returncode == 0
    content = output.read_text(encoding="utf-8-sig")
    assert content.splitlines()[0] == "a;b"


# =====================================================================
# Порядок операций в clean
# =====================================================================

def test_cli_clean_dedup_runs_after_normalization(tmp_path):
    """Дедупликация должна выполняться после trim/normalize — раньше
    она шла первой и не ловила дубли, различавшиеся только пробелами
    или форматом числа/даты."""
    input_file = tmp_path / "in.csv"
    input_file.write_text(
        "id,name,amount\n"
        "1, Иванов ,\"1 200,50\"\n"
        "1,Иванов,1200.50\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.csv"
    result = run_cli("clean", str(input_file), "-o", str(output), "--all")
    assert result.returncode == 0
    df = pd.read_csv(output, dtype=str)
    assert len(df) == 1


# =====================================================================
# split: коллизии имён и NaN
# =====================================================================

def test_cli_split_nan_group_named_without_value(tmp_path):
    input_file = tmp_path / "in.csv"
    input_file.write_text("id,category\n1,A\n2,\n", encoding="utf-8")
    out_dir = tmp_path / "split_out"
    result = run_cli("split", str(input_file), "--by", "category", "-o", str(out_dir))
    assert result.returncode == 0
    files = sorted(p.name for p in out_dir.iterdir())
    assert f"{input_file.stem}_без_значения.csv" in files


def test_cli_split_filename_collision_gets_suffix(tmp_path):
    input_file = tmp_path / "in.csv"
    input_file.write_text("id,category\n1,A.B\n2,A_B\n3,C\n", encoding="utf-8")
    out_dir = tmp_path / "split_out"
    result = run_cli("split", str(input_file), "--by", "category", "-o", str(out_dir))
    assert result.returncode == 0
    files = sorted(p.name for p in out_dir.iterdir())
    assert f"{input_file.stem}_A_B.csv" in files
    assert f"{input_file.stem}_A_B_2.csv" in files


def test_cli_split_three_way_collision_keeps_all_groups(tmp_path):
    """A/B и A_B оба нормализуются в "A_B", получая суффикс "_2". Если
    третья группа изначально называется "A_B_2", она должна получить
    свой собственный свободный суффикс, а не перезаписать вторую группу."""
    input_file = tmp_path / "in.csv"
    input_file.write_text(
        "id,category\n1,A/B\n2,A_B\n3,A_B_2\n4,C\n", encoding="utf-8",
    )
    out_dir = tmp_path / "split_out"
    result = run_cli("split", str(input_file), "--by", "category", "-o", str(out_dir))
    assert result.returncode == 0
    files = sorted(p.name for p in out_dir.iterdir())
    stem = input_file.stem
    assert len(files) == 4
    assert f"{stem}_A_B.csv" in files
    assert f"{stem}_A_B_2.csv" in files
    assert f"{stem}_A_B_2_2.csv" in files
    assert f"{stem}_C.csv" in files


# =====================================================================
# write_table: защита от инъекции формул в xlsx
# =====================================================================

@pytest.mark.parametrize("raw", ["=1+1", "+1+1", "-1+1", "@SUM(A1)"])
def test_write_table_xlsx_escapes_formula_like_text(tmp_path, raw):
    """Текстовое значение, начинающееся с =, +, -, @, должно записываться
    как текст с префиксом апострофа (не как формула) — иначе комментарий
    или название товара из CSV превращается в исполняемую формулу Excel."""
    df = pd.DataFrame({"comment": [raw]})
    output = tmp_path / "out.xlsx"
    clean_data.write_table(df, str(output))
    wb = openpyxl.load_workbook(output)
    cell = wb.active["A2"]
    assert cell.data_type != "f"
    assert cell.value == f"'{raw}"


def test_write_table_xlsx_does_not_escape_normal_text(tmp_path):
    df = pd.DataFrame({"comment": ["обычный текст"]})
    output = tmp_path / "out.xlsx"
    clean_data.write_table(df, str(output))
    wb = openpyxl.load_workbook(output)
    cell = wb.active["A2"]
    assert cell.value == "обычный текст"


# =====================================================================
# Приведение чисел к числовому типу перед записью в xlsx/json
# =====================================================================

def test_cli_clean_xlsx_output_has_real_numeric_cells(tmp_path):
    input_file = tmp_path / "in.csv"
    input_file.write_text("id,amount\n1,\"1 200,50\"\n", encoding="utf-8")
    output = tmp_path / "out.xlsx"
    result = run_cli(
        "clean", str(input_file), "-o", str(output),
        "--normalize-numbers", "--number-columns", "amount",
    )
    assert result.returncode == 0
    wb = openpyxl.load_workbook(output)
    ws = wb.active
    cell = ws.cell(row=2, column=2)
    assert cell.data_type == "n"
    assert cell.value == pytest.approx(1200.50)


def test_cli_clean_json_output_has_real_numeric_values(tmp_path):
    input_file = tmp_path / "in.csv"
    input_file.write_text("id,amount\n1,\"2 500\"\n", encoding="utf-8")
    output = tmp_path / "out.json"
    result = run_cli(
        "clean", str(input_file), "-o", str(output), "--normalize-numbers",
    )
    assert result.returncode == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(data[0]["amount"], (int, float))
    assert data[0]["amount"] == 2500


def test_cli_clean_csv_output_keeps_numbers_as_text(tmp_path):
    """Для CSV числа остаются текстом (с точкой) — приведение к
    числовому типу нужно только для xlsx/json."""
    input_file = tmp_path / "in.csv"
    input_file.write_text("id,amount\n1,\"1 200,50\"\n", encoding="utf-8")
    output = tmp_path / "out.csv"
    result = run_cli(
        "clean", str(input_file), "-o", str(output),
        "--normalize-numbers", "--number-columns", "amount",
    )
    assert result.returncode == 0
    content = output.read_text(encoding="utf-8-sig")
    assert "1200.50" in content


# =====================================================================
# read_table: xlsx с ошибками чтения
# =====================================================================

def test_read_table_rejects_file_with_wrong_encoding(tmp_path):
    """UTF-16-файл, прочитанный как utf-8 или cp1251, даёт мусорные
    управляющие символы (NUL) вместо понятной ошибки — раньше это
    завершалось кодом 0 с повреждёнными данными."""
    bad_file = tmp_path / "encoding-utf-16.csv"
    bad_file.write_bytes("id,name\n1,Иванов\n".encode("utf-16"))
    result = run_cli("convert", str(bad_file), str(tmp_path / "out.json"))
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in combined
    assert "кодировк" in combined.lower()


def test_cli_convert_invalid_output_path_fails_cleanly(tmp_path):
    """Путь вывода с несуществующим родителем-файлом (не каталогом)
    должен давать понятную ошибку, а не FileExistsError с трейсбеком."""
    input_file = tmp_path / "in.csv"
    input_file.write_text("a;b\n1;2\n", encoding="utf-8")
    bad_output = input_file / "out.csv"
    result = run_cli("convert", str(input_file), str(bad_output))
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in combined
    assert "Ошибка" in combined


def test_cli_convert_invalid_out_sep_fails_cleanly(tmp_path):
    """Многосимвольный --out-sep должен давать понятную ошибку, а не
    TypeError из pandas с трейсбеком."""
    input_file = tmp_path / "in.csv"
    input_file.write_text("a;b\n1;2\n", encoding="utf-8")
    output = tmp_path / "out.csv"
    result = run_cli("convert", str(input_file), str(output), "--out-sep", "::")
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in combined
    assert "Ошибка" in combined


def test_read_table_corrupt_xlsx_fails_cleanly(tmp_path):
    bad_file = tmp_path / "bad.xlsx"
    bad_file.write_text("this is not a real xlsx file", encoding="utf-8")
    result = run_cli("convert", str(bad_file), str(tmp_path / "out.csv"))
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in combined
    assert "Ошибка" in combined


# =====================================================================
# sample_data/orders_export.xlsx — настоящие даты и числа
# =====================================================================

def test_sample_orders_export_has_real_date_and_number_cells():
    path = PROJECT_ROOT / "sample_data" / "orders_export.xlsx"
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    header = [c.value for c in ws[1]]
    date_col = header.index("Дата") + 1
    amount_col = header.index("Сумма") + 1

    date_cell = ws.cell(row=2, column=date_col)
    amount_cell = ws.cell(row=2, column=amount_col)
    assert date_cell.is_date
    assert amount_cell.data_type == "n"


def test_sample_orders_export_readable_and_cleanable(tmp_path):
    path = PROJECT_ROOT / "sample_data" / "orders_export.xlsx"
    output = tmp_path / "clean.csv"
    result = run_cli("clean", str(path), "-o", str(output), "--all")
    assert result.returncode == 0
    assert "Traceback" not in (result.stdout + result.stderr)


# =====================================================================
# CLI
# =====================================================================

def test_cli_clean_missing_input_file_fails_cleanly(tmp_path):
    output = tmp_path / "out.csv"
    result = run_cli("clean", str(tmp_path / "no_such_file.csv"), "-o", str(output), "--all")
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "не найден" in combined
    assert "Traceback" not in combined


def test_cli_split_missing_column_lists_available_columns(tmp_path):
    input_file = tmp_path / "data.csv"
    input_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    result = run_cli(
        "split", str(input_file),
        "--by", "НетТакогоСтолбца",
        "-o", str(tmp_path / "split_out"),
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "не найден" in combined
    assert "a" in combined and "b" in combined
    assert "Traceback" not in combined


def test_cli_merge_mismatched_columns_fails(tmp_path):
    file1 = tmp_path / "f1.csv"
    file2 = tmp_path / "f2.csv"
    file1.write_text("a,b\n1,2\n", encoding="utf-8")
    file2.write_text("a,c\n1,2\n", encoding="utf-8")

    result = run_cli("merge", str(file1), str(file2), "-o", str(tmp_path / "merged.csv"))
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in combined
