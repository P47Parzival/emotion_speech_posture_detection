import argparse, os, sys, glob, re, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import dlib

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.features.landmarks import LandmarkExtractor
from src.features.geometry import compute_geo28_features, polygon_area, DISTANCE_FEATURES_SPEC
from src.utils.labels import infer_label_from_path, EMOTION_ORDER, EMOTION_MAP, label_to_idx, idx_to_label

def flatten_landmarks(pts68):
    # pts68: (68,2) -> (136,)
    return pts68.reshape(-1)

def zero_out_jaw(pts68):
    # Set jaw (points 0..16) to zeros
    pts = pts68.copy()
    pts[0:17,:] = 0.0
    return pts

def make_features(image_paths, predictor_path, upsample=0, features='combo164', drop_jaw=False):
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_path)
    LEX = LandmarkExtractor(face_detector, landmark_predictor, upsample=upsample)

    X_list, y_list, used_paths = [], [], []
    failed_detection = 0

    for p in tqdm(image_paths, desc="Extracting"):
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pts68 = LEX.detect_landmarks(gray)
        if pts68 is None:
            failed_detection += 1
            continue

        if drop_jaw:
            pts68 = zero_out_jaw(pts68)

        # Landmarks features (136)
        lm136 = flatten_landmarks(pts68)

        # Geo-28
        geo28 = compute_geo28_features(pts68)

        if features == 'landmarks68':
            feats = lm136
        elif features == 'geo28':
            feats = geo28
        elif features == 'combo164':
            feats = np.concatenate([lm136, geo28], axis=0)
        else:
            raise ValueError("features must be one of: landmarks68, geo28, combo164")

        label = infer_label_from_path(p)
        if label is None:
            continue

        X_list.append(feats)
        y_list.append(label_to_idx(label))
        used_paths.append(p)

    X = np.vstack(X_list) if X_list else np.zeros((0, 164 if features=='combo164' else (136 if features=='landmarks68' else 28)))
    y = np.array(y_list)
    print(f"Extracted {len(X_list)} samples ({failed_detection} failed face detection)")
    return X, y, used_paths

def get_model(name: str):
    name = name.lower()
    if name == 'svm_linear':
        return 'svm_linear', SVC(kernel='linear', decision_function_shape='ovr')
    if name == 'svm_rbf':
        return 'svm_rbf', SVC(kernel='rbf', decision_function_shape='ovr')
    if name == 'svm_poly':
        return 'svm_poly', SVC(kernel='poly', degree=3, decision_function_shape='ovr')
    if name == 'logreg_l1':
        # saga supports l1 for multinomial/ovr; we'll use ovr to mirror paper
        return 'logreg_l1', LogisticRegression(penalty='l1', solver='saga', max_iter=5000, multi_class='ovr')
    if name == 'logreg_l2':
        return 'logreg_l2', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, multi_class='ovr')
    if name == 'linearsvc_ovr':
        return 'linearsvc_ovr', LinearSVC()  # OVR by default
    if name == 'linearsvc_crammer':
        return 'linearsvc_crammer', LinearSVC(multi_class='crammer_singer')
    if name == 'rf':
        return 'rf', RandomForestClassifier(n_estimators=300, random_state=42)
    if name == 'dt':
        return 'dt', DecisionTreeClassifier(random_state=42)
    raise ValueError(f"Unknown model: {name}")

def run(models, X, y, cv, save_outputs, outdir):
    os.makedirs(outdir, exist_ok=True)
    results = []

    # Get unique labels present in y
    unique_labels = sorted(np.unique(y))
    present_emotions = [idx_to_label(i) for i in unique_labels]
    
    for mname in models:
        tag, model = get_model(mname)
        print(f"\n=== Model: {tag} ===")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # CV score (accuracy)
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Cross-val predict for confusion matrix & classification report
        y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=None)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=unique_labels)
        report = classification_report(y, y_pred, labels=unique_labels, target_names=present_emotions, digits=4)

        print(f"Full-dataset (via cross_val_predict) Accuracy: {acc:.4f}")
        print(report)

        res = {
            "model": tag,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "cv_folds": cv,
            "full_accuracy": float(acc)
        }
        results.append(res)

        if save_outputs:
            pd.DataFrame(cm, index=present_emotions, columns=present_emotions).to_csv(os.path.join(outdir, f"cm_{tag}.csv"))
            with open(os.path.join(outdir, f"report_{tag}.txt"), "w") as f:
                f.write(report)
            with open(os.path.join(outdir, f"scores_{tag}.json"), "w") as f:
                json.dump(res, f, indent=2)

    if save_outputs:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(outdir, "summary.csv"), index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path to images (folder-per-class or files)")
    ap.add_argument("--predictor_path", required=True, help="Path to dlib 68 landmark predictor .dat")
    ap.add_argument("--upsample", type=int, default=0, help="Face detector upsample (0..2)")
    ap.add_argument("--features", default="combo164", choices=["landmarks68","geo28","combo164"])
    ap.add_argument("--drop_jaw", action="store_true", help="Zero-out jaw landmarks (0..16)")
    ap.add_argument("--cv", type=int, default=5, help="Stratified K-Folds")
    ap.add_argument("--models", nargs="+", default=["logreg_l2", "svm_linear"])
    ap.add_argument("--save_outputs", action="store_true")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    # Collect image paths
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    paths = []
    for root, _, files in os.walk(args.data_root):
        for fn in files:
            if fn.lower().endswith(exts):
                paths.append(os.path.join(root, fn))
    if not paths:
        print("No images found under", args.data_root)
        sys.exit(1)

    X, y, used_paths = make_features(paths, args.predictor_path, upsample=args.upsample, features=args.features, drop_jaw=args.drop_jaw)
    if X.shape[0] == 0:
        print("No samples with landmarks & labels extracted. Check data & filenames.")
        sys.exit(1)

    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    run(args.models, X, y, args.cv, args.save_outputs, args.outdir)

if __name__ == "__main__":
    main()
