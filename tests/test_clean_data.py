"""
Тесты для clean_data.py.

Функции очистки тестируются напрямую через DataFrame, без запуска CLI.
Отдельно — несколько CLI-тестов через subprocess для проверки кодов
возврата и текста ошибок.
"""

import subprocess
import sys
from pathlib import Path

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
    result, changed, columns = clean_data.normalize_dates(df, columns=["date"])
    assert result["date"].iloc[0] == expected
    assert columns == ["date"]


def test_normalize_dates_non_date_value_unchanged():
    df = pd.DataFrame({"date": ["не дата"]})
    result, changed, columns = clean_data.normalize_dates(df, columns=["date"])
    assert result["date"].iloc[0] == "не дата"
    assert changed == 0


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
    result, changed, columns = clean_data.normalize_numbers(df, columns=["amount"])
    assert result["amount"].iloc[0] == expected


def test_normalize_numbers_non_numeric_value_unchanged():
    df = pd.DataFrame({"amount": ["не число"]})
    result, changed, columns = clean_data.normalize_numbers(df, columns=["amount"])
    assert result["amount"].iloc[0] == "не число"
    assert changed == 0


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
