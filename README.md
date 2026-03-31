# C-MAP

**C-MAP (CHI Magnetic Analysis Program)** は QUEST トカマクの CHI プラズマを対象とした磁気面再構成プログラムです。
Taylor 緩和後 ($\lambda = \mathrm{const.}$) を仮定しています。
仮定条件の詳細は T. Motoki, Master thesis (2024) を参照してください。

---

## 動作環境

| 項目 | 要件 |
|------|------|
| OS | macOS (推奨) |
| Python | 3.10 以上 |
| 主要ライブラリ | numpy, scipy, matplotlib, numba, Pillow, scikit-learn, tqdm |

> CHI 実験データを取得するため、QUEST サーバーへのマウントを推奨します。

---

## ファイル構成

```
cmap/
├── main.py                   # エントリポイント (~109行)
├── mfield_sub.py             # 後方互換ラッパー (fitting_bz 等)
├── Parameter.py              # 後方互換エクスポート
├── get_data.py               # QUEST サーバーからの実験データ取得
│
├── cmaplib/                  # コアライブラリパッケージ
│   ├── grid.py               # Grid: R-Z グリッド管理
│   ├── tokamak_config.py     # TokamakConfig: 装置形状・コイル配置
│   ├── electromagnetics.py   # ベクトル化電磁場計算 (cal_vecp_2_grid 等)
│   ├── field_line_tracer.py  # Numba JIT + prange 磁力線追跡
│   ├── greens_function.py    # GreenFunction: A_0 (3.4 GB) 管理・集約
│   ├── solver.py             # EquilibriumSolver: 収束ループ
│   ├── preparation.py        # prepare_time_step: 時刻ステップ前処理
│   ├── results.py            # ResultsManager: CSV 出力管理
│   ├── plotting.py           # プロット関数群
│   └── utils.py              # nint, cd_main 等のユーティリティ
│
└── modules/
    ├── mfile_py.bin          # グリーン関数行列 A_0 バイナリ
    ├── PFdata.csv            # PF / TF コイル電流データ
    ├── ele_posi.csv          # 容器壁マスク
    └── Qvessel.png           # QUEST 容器断面図
```

---

## 入力

- TF / PF コイル電流
- 入射電流 (Gun 電流)
- トロイダル電流
- ピックアップコイル計測 Bz

## 出力

- 2 次元ポロイダル磁束分布（磁気面コンタ）
- 2 次元ポロイダル電流分布
- Z-Bz プロファイル比較

---

## 使い方

```python
# main.py の以下の変数を編集して実行
count     = 53034      # ショット番号
t_ana_arr = [18.740]   # 解析時刻 [ms]

python main.py
```

結果は `test30_#<count>/` 以下に保存されます。

```
test30_#53034/
├── data/          # M_field, J_field, z_Bz CSV
├── img/           # コンタプロット PNG
└── <t_ana>/       # 反復ごとの中間ファイル
```

---

## アーキテクチャ概要

```
main.py
  └─ prepare_time_step()        # 真空場・フィッティング・パラメータ設定
  └─ EquilibriumSolver
       ├─ GreenFunction          # A_0 テンソル集約 (einsum)
       ├─ trace_all_field_lines  # Numba JIT + prange 並列磁力線追跡
       └─ 収束判定・CSV保存
```

---

## パフォーマンス最適化

| 処理 | 手法 | 向上倍率 (目安) |
|------|------|----------------|
| 磁力線追跡 | Numba JIT + prange 並列化 | ~300–600x |
| 真空場計算 | NumPy ブロードキャスト (cal_vecp_2_grid) | ~1000x |
| A_phi 集約 | np.einsum テンソル縮約 | ~100x |
| B 場有限差分 | 配列スライス置換 | ~100x |

---

## 出力例

![Output example](Image/psi_cont_19.450_small.png)

---

## Contact

Takuto MOTOKI: takuto.505521@gmail.com
