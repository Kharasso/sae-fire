import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def load_features(feature_dir: str):
    """
    Load stacked features and metadata CSV from a directory.
    Returns X (num_samples, num_features), y (labels), ids (list of transcript_ids).
    Assumes: .npz files named <feat>_start_end.npz, and meta_start_end.csv.
    """
    npz_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.npz')])
    csv_files = sorted([f for f in os.listdir(feature_dir) if f.startswith('meta') and f.endswith('.csv')])
    X_list, ids_list = [], []
    for npz in npz_files:
        data = np.load(os.path.join(feature_dir, npz))['arr']  # (batch, feat_dim)
        X_list.append(data)
    X = np.vstack(X_list)
    metas = [pd.read_csv(os.path.join(feature_dir, csv)).set_index('transcript_id') for csv in csv_files]
    meta = pd.concat(metas)
    y = meta['label'].values
    ids = meta.index.tolist()
    return X, y, ids


def select_anova(X: np.ndarray, y: np.ndarray, k: int = 1000):
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector


def select_tree(X: np.ndarray, y: np.ndarray, threshold: float = 'median'):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    selector = SelectFromModel(clf, threshold=threshold)
    X_new = selector.fit_transform(X, y)
    return X_new, selector

