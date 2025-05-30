import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from utils import thresholds, get_scores, FIGSIZE


def main(ks, ps, title, save_dir):
    plt.figure(figsize=FIGSIZE)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.ylabel('MCC score')
    plt.xlabel('threshold')
    plt.title(f'Matthews Correlation Coefficient {title}')

    for model, (y, p) in ps.items():
        scores = get_scores(y, p, matthews_corrcoef)
        plt.plot(thresholds, scores, label=model)

    plt.legend(loc="best")
    plt.savefig(f'{save_dir}/mcc.png')
