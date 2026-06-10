from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass(frozen=True)
class ModelOption:
    key: str
    label: str
    task: str
    recommended: bool
    description: str


MODEL_OPTIONS = [
    ModelOption("linear_regression", "線形回帰", "regression", True, "数値の関係を直線で近似します。まず試す基準モデルに向いています。"),
    ModelOption("tree_regression", "決定木回帰", "regression", False, "条件分岐で数値を予測します。非線形の傾向も扱いやすいモデルです。"),
    ModelOption("logistic_regression", "ロジスティック回帰", "classification", True, "カテゴリを分類する基本モデルです。結果を説明しやすい点が特徴です。"),
    ModelOption("tree_classification", "決定木分類", "classification", False, "条件分岐で分類します。どの条件が効いたかを追いやすいモデルです。"),
]


def available_models(task: str) -> list[ModelOption]:
    return [model for model in MODEL_OPTIONS if model.task == task]


def infer_task(objective: str, target_series: pd.Series | None) -> str | None:
    if objective == "数値を予測したい":
        return "regression"
    if objective == "グループやカテゴリを分類したい":
        return "classification"
    if target_series is None:
        return None
    if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique(dropna=True) > 10:
        return "regression"
    return "classification"


def build_model(model_key: str):
    if model_key == "linear_regression":
        return LinearRegression()
    if model_key == "tree_regression":
        return DecisionTreeRegressor(max_depth=5, random_state=42)
    if model_key == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    if model_key == "tree_classification":
        return DecisionTreeClassifier(max_depth=5, random_state=42)
    raise ValueError("未対応のモデルです。")


def feature_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    return (
        pd.DataFrame({"変数": feature_names, "重要度": values})
        .sort_values("重要度", ascending=False)
        .reset_index(drop=True)
    )


def run_analysis(model_key: str, task: str, prepared) -> dict:
    model = build_model(model_key)
    model.fit(prepared.X_train, prepared.y_train)
    pred = model.predict(prepared.X_test)
    result = {
        "task": task,
        "model_key": model_key,
        "model": model,
        "predictions": pd.DataFrame({"実測値": prepared.y_test.reset_index(drop=True), "予測値": pred}),
        "feature_importance": feature_importance(model, prepared.feature_names),
    }
    if task == "regression":
        mse = mean_squared_error(prepared.y_test, pred)
        result["metrics"] = {
            "R2スコア": r2_score(prepared.y_test, pred),
            "MAE": mean_absolute_error(prepared.y_test, pred),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
        }
    else:
        average = "weighted"
        labels = sorted(pd.Series(prepared.y_test).dropna().unique().tolist())
        result["metrics"] = {
            "正解率": accuracy_score(prepared.y_test, pred),
            "適合率": precision_score(prepared.y_test, pred, average=average, zero_division=0),
            "再現率": recall_score(prepared.y_test, pred, average=average, zero_division=0),
            "F1スコア": f1_score(prepared.y_test, pred, average=average, zero_division=0),
        }
        result["confusion_matrix"] = pd.DataFrame(confusion_matrix(prepared.y_test, pred, labels=labels), index=labels, columns=labels)
    return result
