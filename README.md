# データ分析はじめてナビ

データ分析の初心者が、CSVの取り込みから可視化、基本モデルの実行、結果の解釈、現実のアクション検討までを一連の流れで体験できるStreamlitアプリです。

## 主な機能

- 分析目的の選択
- CSVアップロードと文字コード・区切り文字の切り替え
- サンプルCSVのダウンロードと読み込み
- 目的変数・説明変数・除外列の設定
- データ型、欠損値、基本統計量、重複、外れ値候補の確認
- 散布図、棒グラフ、折れ線グラフ、ヒストグラム、箱ひげ図、相関ヒートマップ、円グラフ、カテゴリ別集計グラフ
- 欠損値処理、カテゴリ変数の数値化、標準化、train/test split
- 線形回帰、決定木回帰、ロジスティック回帰、決定木分類
- R2、MAE、MSE、RMSE、正解率、適合率、再現率、F1スコア、混同行列、特徴量重要度の表示
- 初心者向けの用語説明
- アクション検討メモ
- 評価指標CSVとHTMLレポートの出力

## セットアップ

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Codex付属Pythonを使う場合:

```powershell
& 'C:\Users\amhik\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m streamlit run app.py
```

## ファイル構成

```text
app.py
requirements.txt
analyst_app/
  data_io.py          CSV読み込み、データ概要、外れ値候補
  modeling.py         モデル選択、学習、評価指標
  preprocessing.py    欠損値処理、カテゴリ変数化、標準化、分割
  reporting.py        CSV/HTMLレポート出力
  samples.py          サンプルデータ生成
  terms.py            用語説明
  visualization.py    グラフ生成
```

## MVP範囲

初期版では、要求事項のMVPに合わせて、CSVアップロード、サンプルデータ、変数設定、データ確認、基本可視化、前処理、線形回帰、ロジスティック回帰、決定木、結果表示、用語説明、簡易レポート出力を実装しています。
