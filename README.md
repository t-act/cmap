# CMAP

## Instruction
C-MAP (CHI Magnetic Analysis Program) はCHIプラズマの磁気面再構成に使用される．
Taylor 緩和後($\lambda=const.$) を仮定している．
仮定条件については T.Motoki, Master thesis (2024). を参照．

### Recommended environment
OS : mac os 
Version : Python 3.9.6
Editor : VS code
> CHI 実験結果を利用するため，QUESTサーバーにマウントを推奨

### File configuration
    /C-MAP
    │
    ├── main.py             # Main file
    ├── mfield_sub.py       # Function file
    ├── Parameter.py        # Paramters file
    ├── get_data.py         # Get CHI data from QUEST server
    └── modules
        ├── mfile.bin       # A_phi binary data
        ├── PFcdata.csv     # PF, TF data
        ├── ele_posi.csv    # Electrode position file
        └── Qvessel.png     # QUEST vessel pic.


### Input
- 磁気コイル
- 入射電流
- トロイダル電流
- TF, PF コイル電流

### Output
- 2次元ポロイダル磁束分布（磁気面）
- 2次元ポロイダル電流分布


## Mechanism

C-MAP は Taylor 緩和後の Force-Free 状態を仮定し，$\lambda$ が空間的に一様であるとみなして磁気面再構成を行う．Taylor 状態では磁場は

$$\nabla\times\mathbf{B}=\lambda\mathbf{B},\qquad \mu_0\mathbf{j}=\lambda\mathbf{B}$$

を満たし，$\mathbf{j}\parallel\mathbf{B}$（$\mathbf{j}\times\mathbf{B}=0$）の Force-Free となる．$\lambda$ はプラズマ全体の積分量に対して定義されるため，緩和状態では空間的に一定になる（詳細は T. Motoki, Master thesis (2024) 2.2 節）．

### 計算手順

1. **入力値の設定**：トロイダル電流値 $I_\mathrm{tor}$，真空容器壁への入射電流値 $I_\mathrm{inj\_wall}$，TF コイル電流 $I_\mathrm{TF}$，PF コイル電流 $I_\mathrm{PF}$，磁気コイルが計測した CS 上の $B_z$ を入力する．$I_\mathrm{inj\_wall}$ は高磁場側に設置した 8 個の磁気センサ（番号 3–10）を用いて Lower CS・CS・Top wall の 3 区分に分けて入力する．

2. **ベクトルポテンシャルの計算**：$I_\mathrm{tor}$ を 9 等分してフィラメント電流（正方形 9 点）を仮定し自己場を計算，$I_\mathrm{PF}$ が作る真空磁場と重ね合わせてベクトルポテンシャル $A_\phi$ を求める．

3. **$\lambda$ の決定**：$A_\phi$ からポロイダル磁束 $\psi_\mathrm{pol}$ を求め，Force-Free 条件から

$$\lambda=\frac{\mu_0\, j_\mathrm{pol}}{B_\mathrm{pol}}=\frac{\mu_0\, j_{\mathrm{inj\_wall},\mathrm{pol}}}{B_\mathrm{pol}}$$

   により $\lambda$ を算出する．

4. **壁入射電流トロイダル成分の計算**：$A_\phi$ から得たトロイダル磁束 $B_\mathrm{tor}$ と $\lambda$ を用いて，各グリッドの壁への入射電流のトロイダル成分を

$$j_{\mathrm{inj\_wall},\phi}=\frac{1}{\mu_0}\lambda B_\mathrm{tor}$$

   から求める．

5. **新ベクトルポテンシャルの計算**：$j_{\mathrm{inj\_wall},\phi}$ を用いて新しいベクトルポテンシャル $A_{\phi,\mathrm{new}}$ を計算する．

6. **収束判定**：$A_\phi$ と $A_{\phi,\mathrm{new}}$ の誤差が一定以下であり，かつ $A_{\phi,\mathrm{new}}$ から計算した $B_z$ と入力値 $B_z$ の最小二乗誤差が最小値を更新した場合にデータを保存し，$A_{\phi,\mathrm{new}}$ を用いて $\lambda$ を再計算する．

7. **フィラメント位置の移動**：収束条件を満たさない場合は $I_\mathrm{tor}$ のフィラメント電流位置を移動し，$A_{\phi,\mathrm{new}}$ を再計算する．

8. 規定のループ回数に達するまで 3–7 を繰り返す．

出力として，ポロイダル磁束の等高線（磁気面），最外殻磁気面（LCFS）および閉磁気面領域の判定，ポロイダル電流分布が得られる．

> **注意**：本コードは全時刻で $\lambda$ 一様を仮定しており，渦電流の影響も考慮していない．そのため，磁束の時間変化が速いプラズマ発展段階では計測値との差が大きくなる．今後は磁気ヘリシティの時間変化や $\lambda$ の空間分布を計測して組み込む必要がある．

## How to use

### 1. 準備
- QUEST サーバをマウントする（`get_data.py` は `/Volumes/share/...` 以下の実験データを参照する）．
- Python 3.9.6 環境を用意し，依存パッケージを導入する．

```bash
pip install numpy matplotlib scipy scikit-learn Pillow icecream tqdm
```

### 2. 設定（`main.py`）
`if __name__=="__main__":` 内の以下を編集する．

```python
count = 53034            # 解析するショット番号
s = g.get_CHI_Data(count, True)   # 第2引数 switch: mac OS -> True, win OS -> False
```

解析時刻 `t_ana_arr` [ms] を目的に応じて設定する（リスト，または `np.arange` で複数時刻）．

```python
t_ana_arr = [18.740]              # 例：単一時刻
# t_ana_arr = np.arange(19.2, 19.7, 0.005)   # 例：時間掃引
```

> 計算格子は R: 0–2 m，Z: −2–2 m，$dr=dz=0.02$ m で固定．

### 3. 実行

```bash
python main.py
```

初回実行時は相互インダクタンス（Green 関数）のバイナリ `modules/mfile_py.bin` が自動生成される．格子全点の二重ループのため時間がかかるが，2 回目以降はこのファイルを読み込むため高速化される．

### 4. 出力
結果は `test30_#{count}/{t_ana:.3f}/` 以下に保存される（磁気面プロット，ポロイダル磁束・電流の CSV，`_log.csv` など）．出力例は [Example output](#example-output) を参照．

## Example output

![Output_example](/Image/psi_cont_19.450_small.png) 

