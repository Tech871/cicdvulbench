from sklearn.metrics import auc, precision_recall_curve
import matplotlib.pyplot as plt
from utils import FIGSIZE


def main(ks, ps, title, save_dir):
    plt.figure(figsize=FIGSIZE)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.title(f'Precision vs Recall {title}')

    for model, (y, p) in ps.items():
        precision, recall, thresholds = precision_recall_curve(y, p)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{model} = {pr_auc:.3f}')

    plt.legend(loc="best")
    plt.savefig(f'{save_dir}/pr.png')
