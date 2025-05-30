import numpy as np
import matplotlib.pyplot as plt
from utils import thresholds, get_scores, FIGSIZE


def get_score(y, p):
    y = np.array(y)
    tp = np.sum((p == 1) & (y == 1))
    tn = np.sum((p == 0) & (y == 0))
    fp = np.sum((p == 1) & (y == 0))
    fn = np.sum((p == 0) & (y == 1))

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = fn / (tn + fn) if (tn + fn) > 0 else 0

    return ppv - npv


def main(ks, ps, title, save_dir):
    plt.figure(figsize=FIGSIZE)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.ylabel('markedness')
    plt.xlabel('threshold')
    plt.title(f'Markedness {title}')

    for model, (y, p) in ps.items():
        scores = get_scores(y, p, get_score)
        plt.plot(thresholds, scores, label=model)


    plt.legend(loc="best")
    plt.savefig(f'{save_dir}/markedness.png')
