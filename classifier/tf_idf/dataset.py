from sklearn.feature_extraction.text import TfidfVectorizer
from storage import load_csv

import numpy as np

import cityhash


def single_to_multi(x, y):
    hashes = list(map(cityhash.CityHash64, x))
    hr, ir = np.unique(hashes, return_inverse=True)
    ri = np.zeros(len(hr), dtype=int)
    for i, r in enumerate(ir):
        ri[r] = i
    rx = x[ri]
    ry = np.zeros((len(hr), max(y) + 1), dtype=int)
    ry[ir, y] = 1
    return rx, ry


def get_xy(language, target, source, s3=None):
    x = load_csv('code', language, target, source=source, s3=s3)
    y = load_csv('target', language, target, source=source, s3=s3)
    return x, y


def concat_xy(language, targets, source, classifier, s3=None):
    xs = []
    ys = []
    if len(targets) == 1 or classifier.get('num_labels'):
        for target in targets:
            x, y = get_xy(language, target, source, s3)
            xs.append(x)
            ys.append(y)
    else:
        for label, target in enumerate(targets, 1):
            x, y = get_xy(language, target, source, s3)
            xs.append(x)
            ys.append(label * y)

    y = np.concatenate(ys)
    x = np.concatenate(xs)
    return x, y


def load_xy(language, target, source, classifier, s3=None):
    if isinstance(target, int):
        x, y = get_xy(language, target, source, s3)
    elif isinstance(target, list):
        x, y = concat_xy(language, target, source, classifier, s3)

        if classifier.get('problem_type') == 'multi_label_classification':
            x, y = single_to_multi(x, y)
    else:
        raise TypeError

    return x, y


def load_dataset(classifier, language, target, source, vectorizer=None, s3=None):
    x, y = load_xy(language, target, source, classifier, s3)
    if vectorizer:
        return vectorizer.transform(x), y

    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(x)
    return x, y, vectorizer
