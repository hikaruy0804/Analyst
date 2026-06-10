from __future__ import annotations

from datetime import datetime

import pandas as pd


def metrics_to_csv(metrics: dict[str, float]) -> bytes:
    return pd.DataFrame([metrics]).to_csv(index=False).encode("utf-8-sig")


def build_html_report(
    objective: str,
    target: str | None,
    features: list[str],
    model_label: str | None,
    metrics: dict[str, float] | None,
    action_notes: dict[str, str],
) -> str:
    metric_rows = ""
    if metrics:
        metric_rows = "".join(f"<tr><th>{key}</th><td>{value:.4f}</td></tr>" for key, value in metrics.items())
    action_rows = "".join(f"<h3>{key}</h3><p>{value or ''}</p>" for key, value in action_notes.items())
    feature_text = ", ".join(features) if features else "未設定"
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>分析レポート</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.7; max-width: 960px; margin: 32px auto; color: #172033; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef2f7; width: 32%; }}
  </style>
</head>
<body>
  <h1>分析レポート</h1>
  <p>作成日時: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
  <h2>分析設定</h2>
  <table>
    <tr><th>分析目的</th><td>{objective}</td></tr>
    <tr><th>目的変数</th><td>{target or "未設定"}</td></tr>
    <tr><th>説明変数</th><td>{feature_text}</td></tr>
    <tr><th>モデル</th><td>{model_label or "未実行"}</td></tr>
  </table>
  <h2>主な結果</h2>
  <table>{metric_rows or "<tr><td>分析結果はまだありません。</td></tr>"}</table>
  <h2>アクション検討</h2>
  {action_rows}
</body>
</html>"""

