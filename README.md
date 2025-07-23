
# NSL-KDD Dropout Evaluation with MLP

このリポジトリは、NSL-KDDデータセットを用いた侵入検知モデル（MLP）におけるDropout率の違いによる分類性能の変化を分析した実験コードとレポートをまとめたものです。

## 📁 構成

```

.
├── main.py                    # 実行スクリプト
├── model.py                   # MLPモデル定義
├── preprocess.py              # データ前処理ロジック
├── utils.py                   # 各種ユーティリティ
├── fig/                       # 混同行列・学習曲線の図
├── 写真/                      # データ分布の棒グラフ
├── main.tex                   # LaTeXレポート本体
├── README.md                  # 本ファイル
└── requirements.txt           # 依存パッケージ一覧（Poetry使用者向け）

````

## 🧪 実験概要

- 使用データセット：NSL-KDD（KDDTrain+.txt / KDDTest+.txt）
- 分類ラベル：5クラス（正常 + 4種の攻撃）
- モデル構造：MLP（隠れ層 [128, 64], Dropout, ReLU）
- 評価指標：
  - Accuracy
  - Precision / Recall / F1スコア（Macro / Weighted）
  - Confusion Matrix
  - Loss曲線

## 🧰 実行環境

- Python 3.10+
- Poetry 使用（または pip）

### Poetry 環境構築方法

```bash
poetry install
poetry run python main.py
````

もしくは、pip使用の場合：

```bash
pip install -r requirements.txt
python main.py
```

## 📊 Dropout実験

以下のDropout率で比較実験を実施：

| Dropout | Accuracy    | Macro‑F1  | Weighted‑F1 |
| ------- | ----------- | --------- | ----------- |
| 0.0     | 88.23 %     | 0.864     | 0.882       |
| 0.3     | **90.47 %** | **0.889** | **0.904**   |
| 0.5     | 89.71 %     | 0.880     | 0.897       |
| 0.7     | 87.05 %     | 0.855     | 0.871       |
| 0.9     | 83.12 %     | 0.811     | 0.827       |

---

## Troubleshooting

* **ModuleNotFoundError** → run `pip install -r requirements.txt` inside the activated virtual environment.
* **CUDA Out of Memory** → reduce batch size (`--batch_size 64`) or train on CPU.
* **Different NumPy / Pandas versions** → the code is tested on *Python 3.11*, *PyTorch 2.3*, *NumPy 1.26*, *Pandas 2.2*.

---

## License

This project is released under the **MIT License**.  Feel free to use, modify and cite.
