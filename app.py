from __future__ import annotations

import pandas as pd
import streamlit as st

from analyst_app.data_io import detect_column_groups, outlier_summary, profile_dataframe, read_csv_flexible
from analyst_app.modeling import available_models, infer_task, run_analysis
from analyst_app.preprocessing import prepare_for_model
from analyst_app.reporting import build_html_report, metrics_to_csv
from analyst_app.samples import get_sample_datasets
from analyst_app.terms import get_term_rows
from analyst_app.visualization import GRAPH_HELP, confusion_matrix_chart, feature_importance_chart, make_chart, prediction_chart


st.set_page_config(page_title="データ分析はじめてナビ", page_icon="📊", layout="wide")


OBJECTIVES = [
    "数値を予測したい",
    "グループやカテゴリを分類したい",
    "データの傾向を可視化したい",
    "どのような分析をすべきか検討したい",
    "まだ目的が明確ではない",
]

MISSING_STRATEGIES = ["欠損がある行を削除", "平均値で補完", "中央値で補完", "最頻値で補完"]


def init_state() -> None:
    defaults = {
        "df": None,
        "source_name": None,
        "objective": OBJECTIVES[0],
        "target": None,
        "features": [],
        "exclude_cols": [],
        "id_cols": [],
        "date_cols": [],
        "missing_strategy": "中央値で補完",
        "encode_categories": True,
        "standardize": False,
        "test_size": 0.25,
        "model_key": None,
        "analysis_result": None,
        "action_notes": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def current_df() -> pd.DataFrame | None:
    return st.session_state.get("df")


def select_sample(sample_key: str) -> None:
    sample = next(item for item in get_sample_datasets() if item.key == sample_key)
    st.session_state.df = sample.dataframe.copy()
    st.session_state.source_name = sample.name
    st.session_state.objective = sample.purpose
    groups = detect_column_groups(st.session_state.df)
    st.session_state.target = "売上" if "売上" in st.session_state.df.columns else st.session_state.df.columns[-1]
    st.session_state.exclude_cols = groups["id"]
    st.session_state.id_cols = groups["id"]
    st.session_state.date_cols = groups["date"]
    st.session_state.features = [col for col in st.session_state.df.columns if col != st.session_state.target and col not in groups["id"]]
    st.session_state.analysis_result = None


def guide(text: str) -> None:
    st.info(text, icon="💡")


def require_data() -> pd.DataFrame | None:
    df = current_df()
    if df is None:
        st.warning("先に「データ取り込み」タブでCSVをアップロードするか、サンプルCSVを読み込んでください。")
    return df


def metric_cards(metrics: dict[str, float]) -> None:
    cols = st.columns(min(len(metrics), 4))
    for idx, (name, value) in enumerate(metrics.items()):
        cols[idx % len(cols)].metric(name, f"{value:.4f}")


init_state()

st.title("データ分析はじめてナビ")
st.caption("CSVの取り込みから可視化、基本モデルの実行、結果の解釈、次のアクション整理までを一つの流れで進めます。")

tabs = st.tabs(
    [
        "1. はじめに",
        "2. データ取り込み",
        "3. データ設定",
        "4. データ確認",
        "5. 可視化",
        "6. 前処理",
        "7. モデル選択",
        "8. 分析結果",
        "9. 結果の解釈",
        "10. アクション検討",
        "11. レポート出力",
    ]
)

with tabs[0]:
    guide("まず分析目的を選びます。目的があいまいな場合は、可視化やデータ確認から始める設定を選んでください。")
    st.session_state.objective = st.radio("分析目的", OBJECTIVES, index=OBJECTIVES.index(st.session_state.objective), horizontal=True)
    if st.session_state.objective == "数値を予測したい":
        st.success("おすすめ: 目的変数が数値なら、まず線形回帰で基準を作り、必要に応じて決定木回帰を試します。")
    elif st.session_state.objective == "グループやカテゴリを分類したい":
        st.success("おすすめ: 目的変数がカテゴリなら、まずロジスティック回帰で分類を試します。")
    else:
        st.success("おすすめ: まずデータ確認と可視化で、使えそうな列や気になる傾向を探します。")

    st.subheader("サンプルCSV")
    st.write("手元にCSVがない場合は、サンプルを読み込んで操作を試せます。")
    for sample in get_sample_datasets():
        with st.expander(sample.name, expanded=False):
            st.write(sample.description)
            st.caption(f"想定目的: {sample.purpose}")
            st.dataframe(pd.DataFrame({"列名": sample.columns.keys(), "意味": sample.columns.values()}), width="stretch")
            col1, col2 = st.columns([1, 1])
            col1.download_button(
                "CSVをダウンロード",
                data=sample.to_csv_bytes(),
                file_name=f"{sample.key}.csv",
                mime="text/csv",
                key=f"download_{sample.key}",
            )
            if col2.button("このサンプルを読み込む", key=f"load_{sample.key}"):
                select_sample(sample.key)
                st.rerun()

with tabs[1]:
    guide("CSVをアップロードすると、先頭行、列名、データ型、欠損数を確認できます。文字化けや列が1つにまとまる場合は文字コードや区切り文字を変えてください。")
    uploaded = st.file_uploader("CSVファイル", type=["csv"])
    col1, col2 = st.columns(2)
    encoding = col1.selectbox("文字コード", ["自動", "utf-8-sig", "utf-8", "cp932", "shift_jis"])
    separator_label = col2.selectbox("区切り文字", ["自動", "カンマ", "タブ", "セミコロン"])
    separator_map = {"自動": "自動", "カンマ": ",", "タブ": "\t", "セミコロン": ";"}
    if uploaded:
        try:
            df = read_csv_flexible(uploaded, encoding=encoding, separator=separator_map[separator_label])
            if len(df) > 100_000:
                st.warning("10万行を超えるCSVです。画面表示や可視化では一部をサンプリングします。")
            st.session_state.df = df
            st.session_state.source_name = uploaded.name
            st.session_state.analysis_result = None
            st.success(f"{uploaded.name} を読み込みました。")
        except ValueError as exc:
            st.error(str(exc))
    df = current_df()
    if df is not None:
        st.subheader("先頭データ")
        st.dataframe(df.head(20), width="stretch")
        st.subheader("列の概要")
        st.dataframe(profile_dataframe(df), width="stretch")

with tabs[2]:
    df = require_data()
    if df is not None:
        guide("目的変数は予測・分類したい列です。説明変数は、その判断に使う材料の列です。IDやメモ列は除外するのが基本です。")
        groups = detect_column_groups(df)
        st.session_state.target = st.selectbox(
            "目的変数",
            df.columns.tolist(),
            index=df.columns.tolist().index(st.session_state.target) if st.session_state.target in df.columns else len(df.columns) - 1,
            help="売上、購入有無、顧客区分など、知りたい結果の列を選びます。",
        )
        st.session_state.exclude_cols = st.multiselect(
            "分析に使わない列",
            df.columns.tolist(),
            default=[col for col in st.session_state.exclude_cols if col in df.columns],
            help="ID、氏名、自由記述など、予測材料にしにくい列を外します。",
        )
        default_features = [col for col in df.columns if col != st.session_state.target and col not in st.session_state.exclude_cols]
        st.session_state.features = st.multiselect(
            "説明変数",
            [col for col in df.columns if col != st.session_state.target],
            default=[col for col in st.session_state.features if col in df.columns and col != st.session_state.target] or default_features,
            help="目的変数を説明・予測するために使う列を選びます。",
        )
        col1, col2, col3, col4 = st.columns(4)
        st.session_state.id_cols = col1.multiselect("ID列", df.columns.tolist(), default=[col for col in groups["id"] if col in df.columns])
        st.session_state.date_cols = col2.multiselect("日付列", df.columns.tolist(), default=[col for col in groups["date"] if col in df.columns])
        col3.metric("数値列", len(groups["numeric"]))
        col4.metric("カテゴリ列", len(groups["categorical"]))

with tabs[3]:
    df = require_data()
    if df is not None:
        guide("モデルを実行する前に、欠損、重複、外れ値候補を確認します。問題が多い列は前処理や除外を検討してください。")
        col1, col2, col3 = st.columns(3)
        col1.metric("行数", f"{len(df):,}")
        col2.metric("列数", f"{df.shape[1]:,}")
        col3.metric("重複行", f"{df.duplicated().sum():,}")
        st.subheader("列ごとの状態")
        st.dataframe(profile_dataframe(df), width="stretch")
        st.subheader("基本統計量")
        st.dataframe(df.describe(include="all").T, width="stretch")
        st.subheader("外れ値候補")
        outliers = outlier_summary(df)
        st.dataframe(outliers if not outliers.empty else pd.DataFrame({"メッセージ": ["数値列がないため外れ値候補は表示できません。"]}), width="stretch")

with tabs[4]:
    df = require_data()
    if df is not None:
        guide("グラフは分析前の重要な確認です。まず分布、カテゴリ別の差、数値同士の関係を見てください。")
        chart_df = df.sample(5_000, random_state=42) if len(df) > 5_000 else df
        graph_type = st.selectbox("グラフの種類", list(GRAPH_HELP.keys()))
        st.caption(GRAPH_HELP[graph_type])
        numeric_cols = chart_df.select_dtypes(include="number").columns.tolist()
        all_cols = chart_df.columns.tolist()
        col1, col2, col3, col4 = st.columns(4)
        x = col1.selectbox("X軸 / 対象列", all_cols, index=0)
        y_options = numeric_cols or all_cols
        y = col2.selectbox("Y軸", y_options, index=0) if graph_type not in ["ヒストグラム", "円グラフ", "相関ヒートマップ"] else None
        color = col3.selectbox("色分け", ["なし", *all_cols])
        agg = col4.selectbox("集計方法", ["合計", "平均"])
        try:
            fig = make_chart(chart_df, graph_type, x=x, y=y, color=None if color == "なし" else color, agg=agg)
            st.plotly_chart(fig, width="stretch")
        except Exception as exc:
            st.error(f"グラフを作成できませんでした。列の組み合わせを変えてください。原因: {exc}")

with tabs[5]:
    df = require_data()
    if df is not None:
        guide("ここではモデル実行前の最低限の加工を指定します。初心者には、欠損は中央値補完、カテゴリ数値化はオンをおすすめします。")
        st.session_state.missing_strategy = st.selectbox("欠損値の処理", MISSING_STRATEGIES, index=MISSING_STRATEGIES.index(st.session_state.missing_strategy))
        st.session_state.encode_categories = st.checkbox("カテゴリ変数を数値化する", value=st.session_state.encode_categories)
        st.session_state.standardize = st.checkbox("数値データを標準化する", value=st.session_state.standardize)
        st.session_state.test_size = st.slider("テスト用データの割合", min_value=0.1, max_value=0.5, value=float(st.session_state.test_size), step=0.05)
        st.write("現在の説明変数:", ", ".join(st.session_state.features) if st.session_state.features else "未設定")

with tabs[6]:
    df = require_data()
    if df is not None:
        guide("目的に合うモデルだけを表示します。最初はおすすめモデルから試し、結果が弱ければ別モデルを比較します。")
        target_series = df[st.session_state.target] if st.session_state.target in df.columns else None
        task = infer_task(st.session_state.objective, target_series)
        if task is None:
            st.warning("目的がまだ不明確です。可視化で傾向を確認してから、目的変数を設定してください。")
        else:
            models = available_models(task)
            labels = [f"{m.label}（おすすめ）" if m.recommended else m.label for m in models]
            selected_label = st.selectbox("分析モデル", labels)
            selected_model = models[labels.index(selected_label)]
            st.session_state.model_key = selected_model.key
            st.success(f"選択中: {selected_model.label}")
            st.write(selected_model.description)

with tabs[7]:
    df = require_data()
    if df is not None:
        guide("設定した目的変数、説明変数、前処理、モデルを使って分析を実行します。結果は指標とグラフで確認できます。")
        if not st.session_state.target or not st.session_state.features:
            st.warning("先に「データ設定」で目的変数と説明変数を選んでください。")
        else:
            target_series = df[st.session_state.target]
            task = infer_task(st.session_state.objective, target_series)
            model_key = st.session_state.model_key
            if model_key is None and task:
                model_key = available_models(task)[0].key
                st.session_state.model_key = model_key
            if st.button("分析を実行", type="primary"):
                try:
                    with st.spinner("前処理とモデル学習を実行しています..."):
                        prepared = prepare_for_model(
                            df,
                            st.session_state.target,
                            st.session_state.features,
                            st.session_state.missing_strategy,
                            st.session_state.encode_categories,
                            st.session_state.standardize,
                            st.session_state.test_size,
                        )
                        st.session_state.analysis_result = run_analysis(model_key, task, prepared)
                    st.success("分析が完了しました。")
                except Exception as exc:
                    st.error(f"分析を実行できませんでした。設定を見直してください。原因: {exc}")

            result = st.session_state.analysis_result
            if result:
                st.subheader("評価指標")
                metric_cards(result["metrics"])
                st.download_button("評価指標CSVをダウンロード", metrics_to_csv(result["metrics"]), "analysis_metrics.csv", "text/csv")
                st.subheader("予測結果")
                st.dataframe(result["predictions"], width="stretch")
                if result["task"] == "regression":
                    st.plotly_chart(prediction_chart(result["predictions"]), width="stretch")
                elif "confusion_matrix" in result:
                    st.plotly_chart(confusion_matrix_chart(result["confusion_matrix"]), width="stretch")
                if result.get("feature_importance") is not None:
                    st.subheader("重要な説明変数")
                    st.dataframe(result["feature_importance"], width="stretch")
                    st.plotly_chart(feature_importance_chart(result["feature_importance"]), width="stretch")

with tabs[8]:
    guide("指標は単独で良し悪しを決めるものではありません。業務で許容できる誤差や、見逃しと誤検知のどちらが困るかに合わせて見ます。")
    st.dataframe(pd.DataFrame(get_term_rows()), width="stretch")

with tabs[9]:
    guide("分析は結果を見て終わりではありません。何が分かったか、次に何をするかを言葉にしておくと、実務につながります。")
    fields = [
        "分析結果から分かったこと",
        "注目すべき変数",
        "改善対象",
        "実施するアクション",
        "追加で必要なデータ",
        "次回分析したいこと",
    ]
    notes = {}
    for field in fields:
        notes[field] = st.text_area(field, value=st.session_state.action_notes.get(field, ""), height=90)
    st.session_state.action_notes = notes

with tabs[10]:
    guide("設定、主な結果、アクション検討をHTMLレポートとして出力できます。CSVは評価指標の再利用に向いています。")
    result = st.session_state.analysis_result
    model_label = None
    if st.session_state.model_key:
        all_models = {model.key: model.label for model in available_models("regression") + available_models("classification")}
        model_label = all_models.get(st.session_state.model_key)
    metrics = result["metrics"] if result else None
    html = build_html_report(
        st.session_state.objective,
        st.session_state.target,
        st.session_state.features,
        model_label,
        metrics,
        st.session_state.action_notes,
    )
    st.download_button("HTMLレポートをダウンロード", html.encode("utf-8"), "analysis_report.html", "text/html")
    if metrics:
        st.download_button("評価指標CSVをダウンロード", metrics_to_csv(metrics), "analysis_metrics.csv", "text/csv", key="metrics_report")
    st.html(html)
