<div align="center">

# Deep Learning Coursework

Five graduate homework projects in PyTorch: regression on analytic fluid flows, a CNN on CIFAR-10, neural style transfer and a physics-informed neural network. Each task comes in three levels: basic, medium and advanced.

[Flora Liu · floraliu.dev](https://floraliu.dev/about#work)

<img src="docs/style-transfer.gif" width="760" alt="A sea-steps photo taking on Starry Night style, step by step">

</div>

<details open>
<summary><b>English</b></summary>

## Results

**Neural style transfer (HW4).** VGG19 features, optimized with L-BFGS. Both animations come from [`nst_improved.py`](Generative%20and%20Neural%20Style%20Transfer/nst_improved.py), a retrained version of the assignment. Compared with the submitted `dpl4_*.py`, it:

- starts from the photo instead of white noise, so no noise is left in the result
- keeps the photo's aspect ratio instead of squashing it to 512 × 512
- uses the layers from Gatys et al.: content from conv4_2, style from conv1_1 to conv5_1
- adds a small total-variation term, and refines from 384 px to 768 px

<img src="docs/style-zodiac.gif" width="760" alt="A campus photo taking on the style of Mucha's Zodiac, step by step">

The submitted scripts and notebook are unchanged. Sea photo: [Stairs to the sea, Skerries, Ireland](https://www.flickr.com/photos/87690240@N03/16313873296) by Giuseppe Milo, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).

**The other four assignments.** The figures are taken from each write-up:

<img src="docs/results.png" width="760" alt="Couette fit, pipe-flow fit, CIFAR-10 predictions, PINN solution">

## Projects

| # | Folder | Topic | Model |
|---|---|---|---|
| 1 | [`Couette Flow Regression`](Couette%20Flow%20Regression) | Fit linear Couette flow, with and without L2 regularization | Linear regression |
| 2 | [`Hagen-Poiseuille Flow`](Hagen-Poiseuille%20Flow) | Fit the parabolic pipe-flow profile, comparing base, wider, deeper and ReLU models | MLP |
| 3 | [`Convolution`](Convolution) | Classify CIFAR-10 images | 2-layer CNN vs MLP |
| 4 | [`Generative and Neural Style Transfer`](Generative%20and%20Neural%20Style%20Transfer) | Transfer an artwork's style onto photos | VGG19 features |
| 5 | [`PINN`](PINN) | Solve the 1D wave equation with a physics-informed loss | DeepXDE (PyTorch backend) |

Each folder holds `dplN_basic.py`, `dplN_med.py` and `dplN_adv.py`.

## Run

```bash
pip install torch torchvision matplotlib numpy tqdm deepxde
cd "Generative and Neural Style Transfer"
python dpl4_basic.py

# retrained version, and the GIF
python nst_improved.py --content data/images/content/sea-steps.jpg --style "data/images/style/The Starry Night.jpg" --out out/starry --style-weight 1e7
python make_gif.py out/starry data/images/content/sea-steps.jpg "data/images/style/The Starry Night.jpg" "Starry Night" out/starry.gif
```

Paths are resolved relative to each script. CIFAR-10 downloads automatically on the first run.

</details>

<details>
<summary><b>繁體中文</b></summary>

以 PyTorch 完成的五份研究所作業，主題包括解析流場迴歸、CIFAR-10 CNN、神經風格轉換，以及物理資訊神經網路（PINN）。每份作業都分為 basic、medium、advanced 三個難度。

## 成果

**神經風格轉換（HW4）**：使用 VGG19 特徵，以 L-BFGS 最佳化。兩段動畫都來自重新訓練的 [`nst_improved.py`](Generative%20and%20Neural%20Style%20Transfer/nst_improved.py)。和當初繳交的 `dpl4_*.py` 相比：

- 從照片開始最佳化，而不是白雜訊，結果不會殘留雜訊
- 保留照片原比例，不再壓成 512 × 512
- 採用 Gatys 等人的層設定：content 取 conv4_2，style 取 conv1_1 到 conv5_1
- 加上少量 total variation 平滑項，並從 384 px 精修到 768 px

<img src="docs/style-zodiac.gif" width="760" alt="校園照片逐步套上 Mucha《Zodiac》的風格">

當初繳交的程式與 notebook 都保留原樣。海邊照片：[Stairs to the sea, Skerries, Ireland](https://www.flickr.com/photos/87690240@N03/16313873296) by Giuseppe Milo, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)。

**其他四份作業**：圖片取自各份書面報告。

<img src="docs/results.png" width="760" alt="Couette 流擬合、圓管流擬合、CIFAR-10 預測、PINN 解">

## 作業列表

| # | 資料夾 | 主題 | 模型 |
|---|---|---|---|
| 1 | [`Couette Flow Regression`](Couette%20Flow%20Regression) | 擬合線性 Couette 流，比較有無 L2 正則化 | 線性迴歸 |
| 2 | [`Hagen-Poiseuille Flow`](Hagen-Poiseuille%20Flow) | 擬合圓管流的拋物線速度剖面，比較基本、加寬、加深與 ReLU 模型 | MLP |
| 3 | [`Convolution`](Convolution) | CIFAR-10 影像分類 | 2 層 CNN 與 MLP 比較 |
| 4 | [`Generative and Neural Style Transfer`](Generative%20and%20Neural%20Style%20Transfer) | 把畫作的風格套用到照片上 | VGG19 特徵 |
| 5 | [`PINN`](PINN) | 以物理資訊損失函數求解一維波動方程 | DeepXDE（PyTorch 後端） |

每個資料夾內都有 `dplN_basic.py`、`dplN_med.py`、`dplN_adv.py`。

## 執行

```bash
pip install torch torchvision matplotlib numpy tqdm deepxde
cd "Generative and Neural Style Transfer"
python dpl4_basic.py

# retrained version, and the GIF
python nst_improved.py --content data/images/content/sea-steps.jpg --style "data/images/style/The Starry Night.jpg" --out out/starry --style-weight 1e7
python make_gif.py out/starry data/images/content/sea-steps.jpg "data/images/style/The Starry Night.jpg" "Starry Night" out/starry.gif
```

路徑都以程式所在位置為基準。第一次執行時會自動下載 CIFAR-10。

</details>
