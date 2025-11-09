# Facial Emotion Recognition (Paper Replication Starter)

This project replicates the core pipeline from **"Facial Emotion Recognition Using Machine Learning" (Raut, 2018)**:

- Dlib HoG face detector + 68-point landmark predictor
- Feature sets:
  - **landmarks68**: raw 68×(x,y) = 136 features
  - **geo28**: **25 distances** + **3 polygon areas** (eyes + mouth)
  - **combo164**: 136 + 28 = **164** features (matches the paper)
- Optional: **drop-jaw** (zero-out 17 jaw landmarks)
- Models compared:
  - SVM (linear / rbf / poly)
  - Logistic Regression (l1 / l2)
  - LinearSVC (ovr / crammer-singer)
  - RandomForest, DecisionTree
- Stratified K-fold CV, cross_val_predict confusion matrix

## 1) Install

> Python 3.10+ recommended

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

You also need **dlib's 68-point shape predictor** file (e.g. `shape_predictor_68_face_landmarks.dat`). Put it anywhere and supply the path via `--predictor_path`.

## 2) Prepare data

This repo expects **images** in one of these forms:

- **Generic (recommended):** folder-per-class
  ```
  data_root/
    Happy/ img1.jpg ...
    Sadness/ img2.png ...
    Surprise/ ...
  ```
- **CK+ file names:** contain a final `_N` where `N in {1..7}` (e.g., `S137_001_00000014_7.png`)
- **RaFD file names:** contain emotion keywords (e.g., `angry`, `contemptuous`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`).

You can run the same command for either dataset; the loader auto-detects based on names/subfolders.

## 3) Run

Examples:

```bash
# Landmarks only, SVM linear, 5-fold CV; skip jaw features
python -m src.train   --data_root /path/to/images   --predictor_path /path/to/shape_predictor_68_face_landmarks.dat   --features landmarks68 --drop_jaw   --models svm_linear   --cv 5

# Geometry-28 features, multiple models
python -m src.train   --data_root /path/to/images   --predictor_path /path/to/shape_predictor_68_face_landmarks.dat   --features geo28   --models logreg_l1 logreg_l2 svm_linear linearsvc_ovr rf dt   --cv 5

# Combined 164 features, 10-fold, save predictions
python -m src.train   --data_root /path/to/images   --predictor_path /path/to/shape_predictor_68_face_landmarks.dat   --features combo164   --models logreg_l2 linearsvc_crammer   --cv 10 --save_outputs
```

Outputs (reports, confusion matrices, CSVs) will appear in `outputs/`.


python -m src.train `
  --data_root dataset `
  --predictor_path /path/to/shape_predictor_68_face_landmarks.dat `
  --features combo164 `
  --models logreg_l2 svm_linear `
  --cv 5 --save_outputs

  
## 4) Notes / Parity with paper

- Uses **Dlib HOG detector + 68 landmarks**, as in the paper.
- Feature math mirrors the paper’s **25 distances** and **3 polygon areas**.
- **164** feature vector = 136 (landmarks) + 28 (geo).
- CV defaults and metrics match the paper’s comparisons.
- For CK+: you may first rename/copy images so each labeled frame has `_label` suffix as in the thesis.

## 5) Troubleshooting

- If `dlib` is hard to install, try installing precompiled wheels for your OS/Python, or use system packages.
- Face not found? Try `--upsample 1` or `2`, or ensure frontal images.

Good luck!