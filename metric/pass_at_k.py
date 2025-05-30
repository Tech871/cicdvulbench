import matplotlib.pyplot as plt
import numpy as np
from utils import FIGSIZE


def get_scores(k):
    counts = [0] * max(k)
    for i in k:
        counts[i - 1] += 1
    for i in range(len(counts) - 1):
        counts[i + 1] += counts[i]
    s = counts[-1]
    return [100 * c // s for c in counts]


def main(ks, ps, title, save_dir):
    plt.figure(figsize=FIGSIZE)

    plt.ylim([0, 105])

    plt.ylabel('ratio')
    plt.xlabel('k')
    plt.title(f'Pass@k {title}')

    n = 0
    for model, k in ks.items():
        ratios = get_scores(k)
        n = max(n, len(ratios))
        ticks = np.arange(1, len(ratios) + 1)
        plt.plot(ticks, ratios, label=model)

    plt.xticks(np.arange(1, n + 1))
    plt.legend(loc="best")
    plt.savefig(f'{save_dir}/pass_at_k.png')
