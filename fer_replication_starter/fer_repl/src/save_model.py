# save_model.py
import os, joblib, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.train import make_features
from src.utils.labels import EMOTION_ORDER
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--predictor_path", required=True)
ap.add_argument("--features", default="combo164", choices=["landmarks68","geo28","combo164"])
ap.add_argument("--drop_jaw", action="store_true")
ap.add_argument("--upsample", type=int, default=1)
ap.add_argument("--model", default="logreg_l2", choices=["logreg_l2","svm_linear"])
ap.add_argument("--out", default="artifacts/model_combo164_logreg_l2.pkl")
args = ap.parse_args()

# Gather image paths properly
data_root = Path(args.data_root)
if not data_root.exists():
    raise SystemExit(f"Data root does not exist: {data_root}")

image_paths = []
for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
    image_paths.extend(data_root.rglob(f"*{ext}"))
    image_paths.extend(data_root.rglob(f"*{ext.upper()}"))

image_paths = [str(p) for p in image_paths]

if not image_paths:
    raise SystemExit(f"No images found under {data_root}")

print(f"Found {len(image_paths)} images")

os.makedirs("artifacts", exist_ok=True)
X, y, _ = make_features(
    image_paths=image_paths,
    predictor_path=args.predictor_path,
    upsample=args.upsample,
    features=args.features,
    drop_jaw=args.drop_jaw
)

if X.shape[0] == 0:
    raise SystemExit("No samples found. Check paths.")

print(f"Training on {X.shape[0]} samples with {X.shape[1]} features")

if args.model == "logreg_l2":
    clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, multi_class="ovr")
else:
    clf = SVC(kernel="linear", probability=True)  # probability=True for proba

clf.fit(X, y)
joblib.dump(clf, args.out)
print(f"Saved model to: {args.out}")