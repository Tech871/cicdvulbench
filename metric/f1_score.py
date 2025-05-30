import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from utils import thresholds, get_scores, FIGSIZE


def main(ks, ps, title, save_dir):
    plt.figure(figsize=FIGSIZE)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.ylabel('F1 score')
    plt.xlabel('threshold')
    plt.title(f'F1 {title}')

    for model, (y, p) in ps.items():
        scores = get_scores(y, p, f1_score)
        plt.plot(thresholds, scores, label=model)

    plt.legend(loc="best")
    plt.savefig(f'{save_dir}/f1_score.png')
