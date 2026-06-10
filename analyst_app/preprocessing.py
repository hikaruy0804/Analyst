from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    processed_frame: pd.DataFrame


def fill_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    result = df.copy()
    if strategy == "欠損がある行を削除":
        return result.dropna()
    for col in result.columns:
        if not result[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(result[col]):
            if strategy == "平均値で補完":
                result[col] = result[col].fillna(result[col].mean())
            elif strategy == "中央値で補完":
                result[col] = result[col].fillna(result[col].median())
            else:
                result[col] = result[col].fillna(result[col].mode(dropna=True).iloc[0])
        else:
            mode = result[col].mode(dropna=True)
            result[col] = result[col].fillna(mode.iloc[0] if not mode.empty else "不明")
    return result


def prepare_for_model(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    missing_strategy: str,
    encode_categories: bool,
    standardize: bool,
    test_size: float,
    random_state: int = 42,
) -> PreparedData:
    model_df = df[[target, *features]].copy()
    model_df = fill_missing(model_df, missing_strategy)
    if model_df.empty:
        raise ValueError("前処理の結果、分析に使える行がなくなりました。欠損値の処理方法を変更してください。")

    y = model_df[target]
    X = model_df[features]
    if encode_categories:
        X = pd.get_dummies(X, drop_first=False)
    else:
        non_numeric = X.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            raise ValueError("カテゴリ列が含まれています。カテゴリ変数の数値化をオンにしてください。")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 and y.dtype == "object" else None
    )

    if standardize:
        numeric_cols = X_train.select_dtypes(include="number").columns
        scaler = StandardScaler()
        X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

    processed = pd.concat([X, y.rename(target)], axis=1)
    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=X.columns.tolist(),
        processed_frame=processed,
    )

