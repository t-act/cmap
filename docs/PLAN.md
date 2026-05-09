# C-MAP リファクタリング計画

## Context

C-MAP (CHI Magnetic Analysis Program) はQUESTトカマクの磁気フラックス面再構成プログラムで、Fortranから直訳的にPythonへ移植されたもの。手続き型のまま1081行の`main.py`に大半のロジックが集中し、全体像が見えにくい。また、Pythonのforループで数値計算を行っているためパフォーマンスが極めて低い。

**目的**: OOP化により構造を明確にし、NumPyベクトル化+Numba JITにより計算速度を大幅に向上させる。

---

## ファイル構成

```
cmap/
    __init__.py
    grid.py                  # Grid: R-Zメッシュ管理
    tokamak_config.py        # TokamakConfig: コイル・電極・容器形状
    electromagnetics.py      # cal_vecp_2のベクトル化版 + B場計算
    field_line_tracer.py     # Numba JIT化した磁力線追跡
    solver.py                # EquilibriumSolver: 反復計算アルゴリズム
    greens_function.py       # GreenFunction: A_0の管理・計算
    experimental_data.py     # ExperimentalData: 実験データ取得 (get_data.pyリファクタ)
    results.py               # ResultsManager: ファイル出力管理
    plotting.py              # プロット関数 (重複排除)
    utils.py                 # nint, cd_main等のユーティリティ
main.py                      # 薄いエントリポイント (~50行)
modules/                     # データファイル (変更なし)
```

---

## クラス設計

### 1. `Grid` (grid.py)
R-Zグリッドの一元管理。現在`Parameter.py`と`main.py`に分散している定義を統合。

```python
class Grid:
    r, z: np.ndarray          # (101,), (201,)
    R, Z: np.ndarray          # meshgrid (101, 201)
    dr, dz: float             # 0.02
    nr, nz: int               # 101, 201
```

### 2. `TokamakConfig` (tokamak_config.py)
静的な装置形状。`Parameter.py`のデータ + `mfield_sub.py`のcal_sn/get_PF/elect_posiを吸収。

```python
class TokamakConfig:
    elect0, elect: np.ndarray  # 電極位置
    flux_positions: np.ndarray # フラックスループ位置 (118, 2)
    r_c, z_c: np.ndarray      # PFコイル位置 (55,)
    I_pf_c: np.ndarray         # コイル電流
    wq: np.ndarray             # 容器壁マスク (101, 201)

    load_vessel_mask()          # ele_posi.csv読込 (1回のみ)
    load_pf_currents()          # PFdata.csv読込 (1回のみ)
    compute_electrode_normals() # cal_sn()相当
```

### 3. `Electromagnetics` (electromagnetics.py)
電磁場計算のコア。`cal_vecp_2`をベクトル化。

```python
# ベクトル化版: グリッド全体を一括計算
def cal_vecp_2_grid(r_c, z_c, I_pf_c, R_grid, Z_grid) -> (Br, Bz, A_phi)
    # コイル軸(k,1,1)×グリッド(1,nr,nz)のブロードキャスト

# Numbaスカラー版: 磁力線追跡内で使用
@numba.njit
def cal_vecp_2_scalar(...)

# ベクトル化B場計算 (有限差分)
def compute_B_field(A_phi, r, dr, dz) -> (Br, Bz)
    # 配列スライスで一括計算
```

### 4. `FieldLineTracer` (field_line_tracer.py)
最重要ボトルネック。Numba JIT + prange並列化。

```python
@numba.njit
def _trace_single(ir, iz, i_dir, rr10, flux_mag_r, flux_mag_z, A_phi, wq0, wq)
    # 1本の磁力線を最大4000ステップ追跡

@numba.njit(parallel=True)
def trace_all_field_lines(A_phi, flux_mag_r, flux_mag_z, r, z, wq) -> wq_f
    # numba.prangeで全グリッド点を並列処理
    # 各グリッド点ごとにwq0をコピー (現在のループ内コピーと同等)
    # forward+backwardの2方向を追跡
```

### 5. `GreenFunction` (greens_function.py)
Green関数行列A_0 (101x201x101x201) の管理。

```python
class GreenFunction:
    A_0: np.ndarray  # (101, 201, 101, 201), ~3.3GB

    load_or_compute()  # バイナリキャッシュから読込、なければベクトル化版で生成
    accumulate(I_grid, mask) -> A_phi_contrib
        # np.einsum/高度なインデキシングによるテンソル集約
```

### 6. `EquilibriumSolver` (solver.py)
反復計算の主要ロジック。現在の`main.py` L554-1045を吸収。

```python
class EquilibriumSolver:
    __init__(grid, config, green_fn)

    compute_vacuum_field() -> A_phi_0     # PFコイルからの真空場
    solve_time_step(t_ana, ...) -> Result  # 1時刻の反復ソルバー

    # 内部メソッド:
    _compute_B_field(A_phi)
    _trace_field_lines(A_phi, Br, Bz)
    _compute_lambda(wq_f, ...)
    _update_A_phi(A_phi_0, wq_f, lamb, ...)
    _check_convergence(A_phi, A_phi_before)
```

### 7. `Plotting` (plotting.py)
`plot_field`/`plot_field2` → `plot_magnetic_field()` に統合。`plot_z_Bz`/`plot_z_Bz2` → `plot_bz_profile()` に統合。

---

## パフォーマンス最適化 (影響度順)

### P1: 磁力線追跡 (main.py:631-652) — 最大ボトルネック
- **現状**: 20k点 × 2方向 × 4000ステップ = 最大160M回のPythonループ。さらにループ内で`np.copy(wq)` 2万回。
- **対策**: `track_mag_line`を`@numba.njit`化。全グリッド点を`numba.prange`で並列化。wq0コピーをグリッド点ペアごとに1回に削減。
- **期待速度向上**: ~300-600x

### P2: 真空場計算 (main.py:319-323)
- **現状**: 20k回のスカラー版`cal_vecp_2`呼び出し。各呼び出し内で55コイルループ。
- **対策**: `cal_vecp_2_grid()`でコイル軸(55,1,1)×グリッド(1,101,201)をブロードキャストし一括計算。楕円積分の多項式近似はそのまま配列演算化。
- **期待速度向上**: ~1000x

### P3: テンソル集約 (main.py:836-854)
- **現状**: `A_phi_close += I_tor_close * A_0[:,:,ir,iz]`をdoubleループ内で実行。
- **対策**: `np.where`でマスクインデックスを取得し、高度なインデキシング+`np.sum`で一括計算。メモリが厳しい場合はバッチ処理。
- **期待速度向上**: ~100x

### P4: 有限差分B場計算 (main.py:595-602)
- **現状**: 要素ごとのdoubleループ。
- **対策**: 配列スライスで一行に置換: `flux_mag_r = -((A_phi[:-1, 1:] + A_phi[1:, 1:])*0.5 - (A_phi[:-1, :-1] + A_phi[1:, :-1])*0.5)/dz`
- **期待速度向上**: ~100x

### P5: その他の最適化
- `nint()`: `decimal`モジュール → `int(np.floor(value + 0.5))` (~100x高速)
- `deepcopy` → `np.copy()` (main.py:335, 831, 874)
- 時間ループ外へのファイルI/Oホイスト: `wq` (L290), `A_0` (L307), `get_PF()` (L251)
- `sum(J_inj_e[:])`のループ外キャッシュ (L806, 815)
- `np.all([...])` → Python `and` (L670, 684)

---

## 実装順序

各フェーズで参照出力と比較して正しさを保証する。**構造変更と性能最適化を同一ステップで混ぜない**。

### Phase 0: テスト基盤
1. 現状コードを実行し、中間配列 (`A_phi`, `flux_mag_r/z`, `wq_f`, `I_tor`, `lamb`) を`.npy`で保存
2. `tests/`にこれらを参照値とした回帰テストを作成

### Phase 1: ユーティリティ統合 + パッケージ化
- `cmap/`パッケージ作成
- `cmap/utils.py`: `nint()` (高速版), `cd_main()`を統合。`Parameter.py`と`mfield_sub.py`の重複を削除
- **変更ファイル**: `Parameter.py`, `mfield_sub.py`, 新規`cmap/utils.py`
- **検証**: 全体実行、出力一致確認

### Phase 2: データクラス抽出
- `cmap/grid.py`: `Grid`クラス (Parameter.py L187-199 + main.py L172-181)
- `cmap/tokamak_config.py`: `TokamakConfig` (Parameter.py全体 + mfield_sub.pyのget_PF/cal_sn/elect_posi)
- `cmap/experimental_data.py`: `get_data.py`のリネーム・リファクタ
- **変更ファイル**: `Parameter.py`→分解, `mfield_sub.py`から関数移動, `get_data.py`→リファクタ
- **検証**: 全体実行、出力一致確認

### Phase 3: 電磁場計算の抽出 + ベクトル化 (最初の性能向上)
- `cmap/electromagnetics.py`: `cal_vecp_2` → `cal_vecp_2_grid()`(ベクトル化版)
- 真空場doubleループ (main.py:319-323) → 1回の`cal_vecp_2_grid()`呼び出し
- B場有限差分 (main.py:595-602) → 配列スライスに置換
- **変更ファイル**: `mfield_sub.py`から`cal_vecp_2`移動, `main.py`のループ置換
- **検証**: `A_phi`真空場と`flux_mag_r/z`を参照値と比較 (`np.allclose`, rtol=1e-12)

### Phase 4: 磁力線追跡のNumba JIT化
- `cmap/field_line_tracer.py`: `track_mag_line` → `@numba.njit` + `numba.prange`並列化
- `np.copy(wq)`をグリッド点ペアごと1回に削減
- `np.sign` → 明示的条件分岐 (Numba互換)
- **変更ファイル**: `main.py`からtrack_mag_line移動
- **検証**: 全グリッド点の`wq_f`を参照値と完全一致確認

### Phase 5: Green関数・ソルバー抽出
- `cmap/greens_function.py`: A_0のロード・管理、テンソル集約のベクトル化
- `cmap/solver.py`: `EquilibriumSolver` (main.py L554-1045)
- ファイルI/Oの時間ループ外ホイスト
- **変更ファイル**: `main.py`の大部分を`solver.py`へ
- **検証**: 1時刻のソルバー実行、最終`A_phi`と`I_tor`を比較

### Phase 6: 出力・プロット整理
- `cmap/results.py`: `ResultsManager`
- `cmap/plotting.py`: 重複プロット関数の統合
- **変更ファイル**: `mfield_sub.py`のプロット関数移動・統合

### Phase 7: main.pyの再構成
- エントリポイントを~50行のオーケストレータに書き換え

---

## 検証方法

1. **回帰テスト**: 各Phase完了時に参照`.npy`との`np.allclose`比較
2. **収束確認**: 反復ごとのSM値が6桁以上一致
3. **物理的整合性**:
   - 対称コイル配置でA_phiがZ=0対称
   - `sum(I_tor)` ≈ `I_tor_def`
   - 閉磁気面(`wq_f==20`)が連結領域を形成
4. **Numba検証**: JIT追加前後で純Python版と出力完全一致を確認

---

## 対象ファイル一覧

| ファイル | 操作 |
|---------|------|
| `main.py` (1081行) | 大部分を`solver.py`等に分解、最終的に~50行に |
| `mfield_sub.py` (741行) | `electromagnetics.py`, `tokamak_config.py`, `plotting.py`に分散 |
| `Parameter.py` (287行) | `grid.py`, `tokamak_config.py`に分解 |
| `get_data.py` (241行) | `experimental_data.py`にリネーム・リファクタ |
| 新規: `cmap/`パッケージ (9ファイル) | 上記クラス群 |
