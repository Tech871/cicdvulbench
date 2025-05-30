import matplotlib.pyplot as plt
from utils import thresholds, get_scores, FIGSIZE
import numpy as np


def get_score(y, p):
    y = np.array(y).astype(bool)
    p = p.astype(bool)
    return (np.mean(p[y]) + np.mean(~p[~y])) / 2


def main(ks, ps, title, save_dir):
    plt.figure(figsize=FIGSIZE)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.ylabel('accuracy')
    plt.xlabel('threshold')
    plt.title(f'Accuracy {title}')

    for model, (y, p) in ps.items():
        scores = get_scores(y, p, get_score)
        plt.plot(thresholds, scores, label=model)

    plt.legend(loc="best")
    plt.savefig(f'{save_dir}/accuracy.png')
