from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from utils import FIGSIZE


def main(ks, ps, title, save_dir):
    plt.figure(figsize=FIGSIZE)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    plt.title(f'Receiver Operating Characteristic {title}')

    for model, (y, p) in ps.items():
        roc_auc = roc_auc_score(y, p)
        fpr, tpr, thresholds = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f'{model} = {roc_auc:.3f}')

    plt.legend(loc="best")
    plt.savefig(f'{save_dir}/roc.png')
