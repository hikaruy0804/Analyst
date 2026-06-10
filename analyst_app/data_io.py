from __future__ import annotations

from io import BytesIO

import pandas as pd


ENCODINGS = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
SEPARATORS = [",", "\t", ";"]


def read_csv_flexible(uploaded_file, encoding: str = "自動", separator: str = "自動") -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    encodings = ENCODINGS if encoding == "自動" else [encoding]
    separators = SEPARATORS if separator == "自動" else [separator]
    last_error: Exception | None = None
    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(BytesIO(raw), encoding=enc, sep=sep)
                if df.shape[1] > 1 or sep == separators[-1]:
                    return df
            except Exception as exc:  # noqa: PERF203
                last_error = exc
    raise ValueError(
        "CSVを読み込めませんでした。文字コードをcp932やshift_jisに変える、または区切り文字を確認してください。"
    ) from last_error


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "列名": df.columns,
            "データ型": [str(df[col].dtype) for col in df.columns],
            "欠損数": [int(df[col].isna().sum()) for col in df.columns],
            "欠損率": [f"{df[col].isna().mean():.1%}" for col in df.columns],
            "ユニーク数": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )


def detect_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric = df.select_dtypes(include="number").columns.tolist()
    date_like = []
    for col in df.columns:
        name = str(col).lower()
        if "date" in name or "日付" in str(col) or str(col) in {"年月", "月"}:
            date_like.append(col)
    categorical = [col for col in df.columns if col not in numeric and col not in date_like]
    id_like = [col for col in df.columns if "id" in str(col).lower() or "番号" in str(col) or "ID" in str(col)]
    return {"numeric": numeric, "categorical": categorical, "date": date_like, "id": id_like}


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        rows.append({"列名": col, "外れ値候補数": count, "下限目安": lower, "上限目安": upper})
    return pd.DataFrame(rows)

