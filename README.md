# ME5411 CA 2025 – Image Text Pipeline / 图像字符处理流水线

An end-to-end MATLAB pipeline for the coursework:
- Step1–2: display/enhancement + mean filter (own implementations)
- Step3: fixed ROI crop
- Step4: Otsu binarization (own implementation)
- Step5: character outlines
- Step6: CC-based segmentation (+ split-wide to reach ~10 chars)
- Step7: dataset split (75/25), Task1 (CNN), Task2 (HOG+SVM), Task3 (compare)

本仓库给出从原图到字符识别的完整流程：
- 第1–2步：显示/增强 + 可调窗口均值滤波（自研实现）
- 第3步：固定框 ROI 裁剪
- 第4步：Otsu 二值化（自研实现）
- 第5步：字符轮廓提取
- 第6步：连通域分割 + **大框切分**（解决“00”粘连）
- 第7步：数据集 75/25 划分；任务1（CNN），任务2（HOG+SVM），任务3（对比）

---

## 1) Project Structure / 项目结构
```
├─ src/                   # all MATLAB .m files (step1..step7, helpers)
│  ├─ step1_display.m
│  ├─ step1_enhancement.m
│  ├─ step2_filter.m
│  ├─ step3_roi.m
│  ├─ step4_binarize.m
│  ├─ step5_outline.m
│  ├─ step6_segment.m
│  ├─ step7_dataset.m
│  ├─ step7_task1_cnn.m
│  ├─ step7_task1_apply_cnn.m   # simplified framework for teammate to enhance preprocessing/TTA
│  ├─ step7_task2_hogsvm.m
│  ├─ step7_task2_apply_noncnn.m
│  └─ step7_task3_report.m
├─ data/                  # charact2.bmp + dataset_2025.zip (unzipped to data/dataset_2025/)
├─ results/
│  ├─ figures/            # plots & overlays
│  ├─ models/             # cached models: cnn.mat, hogsvm.mat
│  ├─ logs/               # run logs
│  └─ *.csv / *.txt       # predictions & summaries
└─ main.m                 # single entry with switches & caching
```

---

## 2) Requirements / 环境要求
- MATLAB R2020b+（建议）
- Deep Learning Toolbox（Task1: CNN）
- Statistics and Machine Learning Toolbox（Task2: SVM）
- Image Processing Toolbox（推荐；HOG 特征需要）

> 说明：Step1/Step4/Step6 含“无工具箱”实现；Task1/Task2 允许工具箱。

---

## 3) Quick Start / 快速开始
1. Put data under `data/` / 把数据放到 `data/`：
   - `data/charact2.bmp`
   - `data/dataset_2025.zip`（首次运行会自动解压到 `data/dataset_2025/`）
2. Open MATLAB, `cd src`, run / 打开 MATLAB、进入 `src` 目录并运行：
   ```matlab
   main    % run the full pipeline (Steps 1–7) / 运行完整流程
   ```
3. Outputs / 输出：
   - Visualizations: `results/figures/`
   - Models: `results/models/`
   - Prediction tables: `results/*.csv`, `results/*.txt`
   - Whole state snapshot: `results/run_state.mat`

---

## 4) Config & Switches / 配置与开关（编辑 `main.m` 顶部）
```matlab
cfg.flags.runStep1 = true;
cfg.flags.runStep2 = true;
cfg.flags.runStep3 = true;
cfg.flags.runStep4 = true;
cfg.flags.runStep5 = true;
cfg.flags.runStep6 = true;
cfg.flags.runStep7 = true;     % control Step7

% Step7 sub-switches / 子开关
cfg.flags.trainCNN   = true;   % set false to reuse cached models
cfg.flags.trainSVM   = true;
cfg.flags.applyCNN   = true;
cfg.flags.applySVM   = true;
cfg.flags.resumeModels = true; % auto-load from results/models/*
```

Common params / 常用参数：
- **Step3（ROI crop）**：`cfg.roi.bbox = [20 200 950 150];`
- **Step4（二值化）**：`cfg.bin.polarity = 'bright' | 'dark'`
- **Step6（分割）**：
  ```matlab
  cfg.segment.preErodeIters = 1;      % break thin bridges
  cfg.segment.splitWide = true;       % cut overly wide boxes
  cfg.segment.targetK   = 10;         % expect 10 chars
  cfg.segment.splitWidthFactor = 1.35;
  cfg.segment.splitMinPartFrac = 0.45;
  ```
- **Step7（input size）**：`cfg.step7.inputSize = [32 32 1];`

---

## 5) How it works / 关键步骤说明
- **Step1–2**：读取与显示原图、轻度增强；均值滤波窗口可配。  
- **Step3**：按给定 bbox 直接截取 ROI，保证复现实验一致性。  
- **Step4**：自研 Otsu 阈值（支持“亮字/暗字”极性）。  
- **Step5**：从二值图获得轮廓/边缘用于可视化。  
- **Step6**：连通域分析 + 尺寸/宽高比过滤；对超宽候选框执行**一刀切**避免字符粘连。  
- **Step7**：
  - `step7_dataset.m`：自动解压、分层 75/25；
  - **Task1（CNN）**：轻量 LeNet 风格网络（缓存至 `results/models/cnn.mat`）；
  - **Task2（HOG+SVM）**：HOG 特征 + 线性 SVM（缓存至 `results/models/hogsvm.mat`）；
  - **Task3**：导出验证精度、训练/推理耗时、Image1 识别串。

---

## 6) Where to tweak / 常改位置
- **Segmentation**：`src/step6_segment.m`  
  调整 `preErodeIters / minAreaFrac / split*` 影响连通域与大框切分策略。
- **CNN training**：`src/step7_task1_cnn.m`  
  调 `MaxEpochs`、`MiniBatchSize`、`inputSize`，或添加数据增强。
- **CNN apply**：`src/step7_task1_apply_cnn.m`  
  目前为简化框架，方便同学加入 **letterbox/对比度归一/TTA** 等策略。
- **Non-CNN**：`src/step7_task2_hogsvm.m`  
  改 `cellSize`（如 `[8 8]`）、或换核函数（如 `'rbf'`）。

---

## 7) Results & Reports / 结果与报告
- Overlays：`results/figures/step7_task1_cnn_overlay.png`、`step7_task2_hogsvm_overlay.png`  
- Predictions：`results/step7_task*_*.csv` / `*_text.txt`  
- Summary：`results/step7_compare_summary.csv`（验证精度、训练/推理时间、Image1 识别字符串）

---

## 8) Git Tips / Git 使用小贴士
- 首次推送建议先在本地写好 **README** 与 **.gitignore**，避免与远端默认 README 冲突。  
- 建议忽略：`results/`, `logs/`, `data/`, `*.mat`, `*.zip`（见 `.gitignore`）。  
- 协作使用分支 + Pull Request：
  ```bash
  git checkout -b feat/cnn-apply
  # ... edit ...
  git commit -m "Improve CNN apply preprocessing"
  git push -u origin feat/cnn-apply
  ```

---

## 9) License / 许可
默认不附带许可证（课程作业）。如需开放，请在根目录添加 `LICENSE`（例如 MIT）。

---

## 10) Acknowledgements / 致谢
- 部分处理步骤参考课程讲义与 MATLAB 文档；实现与参数均按作业要求做了自定义与限制。
