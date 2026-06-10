from __future__ import annotations

import pandas as pd
import plotly.express as px


GRAPH_HELP = {
    "散布図": "2つの数値項目の関係を見ます。点が右上がりなら、一方が増えるともう一方も増える傾向があります。",
    "棒グラフ": "カテゴリごとの数値比較を見ます。商品別、地域別などの比較に向いています。",
    "折れ線グラフ": "時間や順序に沿った変化を見ます。月別売上などに向いています。",
    "ヒストグラム": "数値データの分布を見ます。多い範囲、偏り、ばらつきを確認できます。",
    "箱ひげ図": "データのばらつきや外れ値候補を見ます。カテゴリ別の分布比較にも使えます。",
    "相関ヒートマップ": "数値項目同士の関係の強さを色で見ます。1に近いほど同じ方向に動きやすい関係です。",
    "円グラフ": "全体に対するカテゴリの割合を見ます。カテゴリ数が少ない場合に向いています。",
    "カテゴリ別集計グラフ": "カテゴリごとに平均や合計を集計して比較します。",
}


def make_chart(df: pd.DataFrame, graph_type: str, x: str | None = None, y: str | None = None, color: str | None = None, agg: str = "合計"):
    if graph_type == "散布図":
        return px.scatter(df, x=x, y=y, color=color, hover_data=df.columns)
    if graph_type == "棒グラフ":
        return px.bar(df, x=x, y=y, color=color)
    if graph_type == "折れ線グラフ":
        return px.line(df.sort_values(x), x=x, y=y, color=color, markers=True)
    if graph_type == "ヒストグラム":
        return px.histogram(df, x=x, color=color, nbins=30)
    if graph_type == "箱ひげ図":
        return px.box(df, x=x, y=y, color=color)
    if graph_type == "相関ヒートマップ":
        corr = df.select_dtypes(include="number").corr(numeric_only=True)
        return px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    if graph_type == "円グラフ":
        counts = df[x].value_counts(dropna=False).reset_index()
        counts.columns = [x, "件数"]
        return px.pie(counts, names=x, values="件数")
    if graph_type == "カテゴリ別集計グラフ":
        grouped = df.groupby(x, dropna=False)[y].agg("sum" if agg == "合計" else "mean").reset_index()
        return px.bar(grouped, x=x, y=y)
    raise ValueError("未対応のグラフ種類です。")


def prediction_chart(predictions: pd.DataFrame):
    plot_df = predictions.reset_index(names="データ番号")
    fig = px.scatter(plot_df, x="実測値", y="予測値", hover_data=["データ番号"])
    values = pd.concat([pd.to_numeric(plot_df["実測値"], errors="coerce"), pd.to_numeric(plot_df["予測値"], errors="coerce")]).dropna()
    if not values.empty:
        lower = values.min()
        upper = values.max()
        fig.add_shape(
            type="line",
            x0=lower,
            y0=lower,
            x1=upper,
            y1=upper,
            line={"color": "#64748b", "dash": "dash"},
        )
        fig.add_annotation(
            x=upper,
            y=upper,
            text="実測値=予測値",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font={"color": "#475569", "size": 12},
        )
    return fig


def feature_importance_chart(importance: pd.DataFrame):
    top = importance.head(20).sort_values("重要度")
    return px.bar(top, x="重要度", y="変数", orientation="h")


def confusion_matrix_chart(matrix: pd.DataFrame):
    return px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", labels={"x": "予測値", "y": "実測値", "color": "件数"})
