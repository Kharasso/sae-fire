import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from feature_selection import select_anova, select_tree, load_features


def train_and_save(X_train, y_train, X_test, out_prob_path, model_type='rf'):
    if model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    elif model_type == 'gb':
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == 'lr':
        clf = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    np.save(out_prob_path, proba)


def main(args):
    X, y, ids = load_features(args.feature_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    if args.selection == 'anova':
        X_train_sel, selector = select_anova(X_train, y_train, k=args.k)
        X_test_sel = selector.transform(X_test)
    elif args.selection == 'tree':
        X_train_sel, selector = select_tree(X_train, y_train, threshold=args.threshold)
        X_test_sel = selector.transform(X_test)
    else:
        X_train_sel, X_test_sel = X_train, X_test

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"proba_{args.selection or 'none'}_{args.model}.npy"
    )
    train_and_save(X_train_sel, y_train, X_test_sel, out_path, model_type=args.model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--selection', choices=['anova', 'tree', 'none'], default='none')
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--threshold', default='median')
    parser.add_argument('--model', choices=['rf', 'gb', 'lr'], default='rf')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
