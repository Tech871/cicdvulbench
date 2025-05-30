import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import thresholds, get_scores, FIGSIZE


def get_score(y, p):
    cm = confusion_matrix(y, p)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    return sensitivity + specificity - 1


def main(ks, ps, title, save_dir):
    plt.figure(figsize=FIGSIZE)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.ylabel('informedness')
    plt.xlabel('threshold')
    plt.title(f'Informedness {title}')

    for model, (y, p) in ps.items():
        scores = get_scores(y, p, get_score)
        plt.plot(thresholds, scores, label=model)

    plt.legend(loc="best")
    plt.savefig(f'{save_dir}/informedness.png')
