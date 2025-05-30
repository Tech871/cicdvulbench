import numpy as np
from storage import load_csv

thresholds = np.arange(0.01, 1.0, 0.01)
FIGSIZE = (10, 10)


def scale_min_max(p):
    m, M = min(p), max(p)
    return np.full(len(p), 0.5) if M == m else np.array([(q - m) / (M - m) for q in p])


def get_predictions(p, t):
    return (p >= t).astype(int)


def get_scores(y, p, get_score):
    p = scale_min_max(p)
    return [get_score(y, get_predictions(p, t)) for t in thresholds]


def get_is_pos(y):
    if len(y.shape) == 1:
        return (y > 0).astype(int)
    return 1 - y[:, 0]


def get_pos_probability(p):
    if len(p.shape) == 1:
        return p
    if p.shape[1] == 1:
        return p[:, 1]
    pos = np.max(p[:, 1:], axis=1)
    neg = p[:, 0]
    pos = np.exp(pos)
    neg = np.exp(neg)
    return pos / (pos + neg)


def get_target_rank(y, p):
    k = []

    if len(p.shape) == 1:
        k = [(y != p) + 1 for y, p in zip(y.astype(bool), p >= 0.5)]

    elif len(y.shape) == 1:
        for y, p in zip(y, p):
            s = np.nonzero(np.sort(p) == p[y])
            k.append(len(p) - s[0].max())

    else:
        for y, p in zip(y.astype(bool), p):
            if np.any(y):
                s = np.nonzero(np.sort(p) == np.max(p[y]))
                k.append(len(p) - s[0].max())
    return k


def get_results(language, target, test, models, s3=None, binary=False):
    ks = {}
    ps = {}

    for model in models:
        train = model.get('train')
        classifier = model['classifier']

        y = load_csv('y', language, target, train=train, test=test, classifier=classifier, s3=s3)
        p = load_csv('p', language, target, train=train, test=test, classifier=classifier, s3=s3)

        model = ' '.join([classifier, train])
        k = get_target_rank(y, p)
        ks[model] = k

        if binary or isinstance(target, int):
            y = get_is_pos(y)
            p = get_pos_probability(p)
            ps[model] = (y, p)
        else:
            for i, t in enumerate(target, 1):
                ps[f'{model} {t}'] = (y[:, i], p[:, i])

    return ks, ps
