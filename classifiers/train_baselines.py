"""
Train baseline classifiers:
 - SAE features without FS: MLP & XGBoost
 - CLS features (last hidden): MLP & LR
"""
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from feature_selection import load_features


def train_and_save(clf, X_train, y_train, X_test, out_path):
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    np.save(out_path, proba)
    print(f"Saved predictions to {out_path}")


def main(args):
    X, y, ids = load_features(args.feature_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    os.makedirs(args.out_dir, exist_ok=True)

    if 'sae' in args.feature_dir:
        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        out_mlp = os.path.join(args.out_dir, 'proba_sae_baseline_mlp.npy')
        train_and_save(mlp, X_train, y_train, X_test, out_mlp)

        xgb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        out_xgb = os.path.join(args.out_dir, 'proba_sae_baseline_xgb.npy')
        train_and_save(xgb, X_train, y_train, X_test, out_xgb)

    if 'cls' in args.feature_dir:
        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        out_mlp = os.path.join(args.out_dir, 'proba_cls_baseline_mlp.npy')
        train_and_save(mlp, X_train, y_train, X_test, out_mlp)

        lr = LogisticRegression(max_iter=1000)
        out_lr = os.path.join(args.out_dir, 'proba_cls_baseline_lr.npy')
        train_and_save(lr, X_train, y_train, X_test, out_lr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
