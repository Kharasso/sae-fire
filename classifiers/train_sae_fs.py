"""
Train logistic regression on SAE features with ANOVA feature importance ranking,
selecting top-K features automatically for specified SAE variants.
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from feature_selection import load_features

# Define K values per SAE variant
K_VALUES = {
    'sae_2b': [500, 1000, 1500, 2000, 2500],
    'sae_9b_131k': [3000, 3500, 4000, 4500, 5000, 5500, 6000],
}


def main(args):
    X, y, ids = load_features(args.feature_dir)
    sae_id = os.path.basename(os.path.normpath(args.feature_dir))
    ks = K_VALUES.get(sae_id)
    if ks is None:
        raise ValueError(f"Unknown SAE variant '{sae_id}' for FS.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for k in ks:
        selector = SelectKBest(f_classif, k=k)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_sel, y_train)
        proba = clf.predict_proba(X_test_sel)[:, 1]
        fname = f"proba_sae_fs_{sae_id}_k{k}_lr.npy"
        out_path = os.path.join(args.out_dir, fname)
        np.save(out_path, proba)
        results.append((k, out_path))
        print(f"Saved predictions for k={k} to {out_path}")

    summary = pd.DataFrame(results, columns=['k','proba_path'])
    summary.to_csv(os.path.join(args.out_dir, 'fs_summary.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-dir', required=True, help='Directory of SAE features')
    parser.add_argument('--out-dir', required=True, help='Output directory for predictions')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()
    main(args)