from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SampleDataset:
    key: str
    name: str
    purpose: str
    description: str
    columns: dict[str, str]
    dataframe: pd.DataFrame

    def to_csv_bytes(self) -> bytes:
        return self.dataframe.to_csv(index=False).encode("utf-8-sig")


def _sales_prediction() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 80
    advertising = rng.integers(20, 220, rows)
    visitors = rng.integers(300, 3500, rows)
    price = rng.integers(800, 3500, rows)
    campaign = rng.choice(["なし", "小規模", "大規模"], rows, p=[0.45, 0.35, 0.20])
    month = rng.integers(1, 13, rows)
    campaign_effect = pd.Series(campaign).map({"なし": 0, "小規模": 120, "大規模": 260}).to_numpy()
    sales = 120 + advertising * 8.5 + visitors * 1.1 - price * 0.18 + campaign_effect + month * 14
    sales = sales + rng.normal(0, 220, rows)
    return pd.DataFrame(
        {
            "月": month,
            "広告費": advertising,
            "来店者数": visitors,
            "平均価格": price,
            "キャンペーン": campaign,
            "売上": np.maximum(sales.round().astype(int), 0),
        }
    )


def _customer_classification() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = 90
    age = rng.integers(18, 72, rows)
    visits = rng.integers(1, 28, rows)
    purchase = rng.integers(1_000, 80_000, rows)
    mail = rng.choice(["未登録", "登録済み"], rows, p=[0.35, 0.65])
    score = visits * 0.18 + purchase / 30_000 + (mail == "登録済み") * 0.8 - age / 95
    segment = np.where(score > 2.4, "優良", "通常")
    return pd.DataFrame(
        {
            "顧客ID": [f"C{i:04d}" for i in range(1, rows + 1)],
            "年齢": age,
            "来店回数": visits,
            "年間購入額": purchase,
            "メール会員": mail,
            "顧客区分": segment,
        }
    )


def _survey_analysis() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    rows = 70
    channel = rng.choice(["店舗", "Web", "紹介", "広告"], rows)
    satisfaction = rng.integers(1, 6, rows)
    usability = rng.integers(1, 6, rows)
    support = rng.integers(1, 6, rows)
    repeat = np.where(satisfaction + usability + support >= 11, "あり", "なし")
    return pd.DataFrame(
        {
            "回答ID": [f"S{i:03d}" for i in range(1, rows + 1)],
            "流入経路": channel,
            "満足度": satisfaction,
            "使いやすさ": usability,
            "サポート評価": support,
            "再利用意向": repeat,
        }
    )


def _product_sales() -> pd.DataFrame:
    rng = np.random.default_rng(101)
    products = ["A商品", "B商品", "C商品", "D商品"]
    rows = []
    for month in range(1, 13):
        for product in products:
            rows.append(
                {
                    "月": month,
                    "商品": product,
                    "販売数量": int(rng.integers(30, 260)),
                    "単価": int(rng.integers(700, 4200)),
                    "地域": rng.choice(["東日本", "西日本", "中部", "九州"]),
                }
            )
    df = pd.DataFrame(rows)
    df["売上"] = df["販売数量"] * df["単価"]
    return df


def get_sample_datasets() -> list[SampleDataset]:
    return [
        SampleDataset(
            key="sales",
            name="売上予測データ",
            purpose="数値を予測したい",
            description="広告費、来店者数、価格などから売上を予測する練習用データです。",
            columns={
                "売上": "予測したい目的変数です。",
                "広告費": "販売促進に使った金額です。",
                "来店者数": "店舗やサイトを訪れた人数です。",
                "平均価格": "商品の平均販売価格です。",
                "キャンペーン": "販促施策の規模を表します。",
            },
            dataframe=_sales_prediction(),
        ),
        SampleDataset(
            key="customers",
            name="顧客分類データ",
            purpose="グループやカテゴリを分類したい",
            description="購買行動から顧客区分を分類する練習用データです。",
            columns={
                "顧客区分": "分類したい目的変数です。",
                "年齢": "顧客の年齢です。",
                "来店回数": "一定期間内の来店回数です。",
                "年間購入額": "年間の購入金額です。",
                "メール会員": "メール会員登録の有無です。",
            },
            dataframe=_customer_classification(),
        ),
        SampleDataset(
            key="survey",
            name="アンケート分析データ",
            purpose="グループやカテゴリを分類したい",
            description="アンケート回答から再利用意向を分類する練習用データです。",
            columns={
                "再利用意向": "分類したい目的変数です。",
                "流入経路": "ユーザーが知った経路です。",
                "満足度": "総合満足度の点数です。",
                "使いやすさ": "サービスの使いやすさの点数です。",
                "サポート評価": "サポート対応への評価です。",
            },
            dataframe=_survey_analysis(),
        ),
        SampleDataset(
            key="products",
            name="商品別売上データ",
            purpose="データの傾向を可視化したい",
            description="商品、地域、月ごとの売上傾向を見る練習用データです。",
            columns={
                "売上": "販売数量と単価から計算した金額です。",
                "商品": "商品カテゴリです。",
                "地域": "販売地域です。",
                "月": "集計対象の月です。",
                "販売数量": "売れた個数です。",
            },
            dataframe=_product_sales(),
        ),
    ]

