import pandas as pd
import numpy as np
import csv
import sys
from typing import Callable

from utils import get_target_rank, get_is_pos, get_pos_probability
from utils import scale_min_max, get_predictions


from accuracy import get_score as get_accuracy
from sklearn.metrics import f1_score as get_f1
from informedness import get_score as get_informedness
from markedness import get_score as get_markedness
from sklearn.metrics import matthews_corrcoef as get_mcc


def get_precision(y: np.ndarray, p: np.ndarray) -> float:
    y = np.array(y)
    tp = np.sum((p == 1) & (y == 1))
    fp = np.sum((p == 1) & (y == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def get_recall(y: np.ndarray, p: np.ndarray) -> float:
    y = np.array(y)
    tp = np.sum((p == 1) & (y == 1))
    fn = np.sum((p == 0) & (y == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def get_fpr(y: np.ndarray, p: np.ndarray) -> float:
    y = np.array(y)
    tn = np.sum((p == 0) & (y == 0))
    fp = np.sum((p == 1) & (y == 0))
    return fp / (fp+tn) if (fp+tn) > 0 else 0


def get_fnr(y: np.ndarray, p: np.ndarray) -> float:
    y = np.array(y)
    tp = np.sum((p == 1) & (y == 1))
    fn = np.sum((p == 0) & (y == 1))
    return fn / (fn+tp) if (fn+tp) > 0 else 0


def get_tpr(y: np.ndarray, p: np.ndarray) -> float:
    y = np.array(y)
    tp = np.sum((p == 1) & (y == 1))
    fn = np.sum((p == 0) & (y == 1))
    return tp / (tp+fn) if (tp+fn) > 0 else 0


FPR_LIMIT = 0.005


def get_vds(y: np.ndarray, p: np.ndarray) -> float:
    fpr = get_fpr(y, p)
    fnr = get_fnr(y, p)
    return fnr if fpr < FPR_LIMIT else 1


def _load_csv(path: str, what: str) -> np.ndarray:
    with open(f'{path}/{what}.csv', encoding='utf-8') as src:
        data = pd.read_csv(src, header=None).values
    return data if data.shape[1] > 1 else data.ravel()


def _get_results(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _load_csv(path, 'y')
    p = _load_csv(path, 'p')
    k = get_target_rank(y, p)
    y = get_is_pos(y)
    p = get_pos_probability(p)
    return (y, p, k)


def _get_cwe_results(path: str, idx: int) -> tuple[np.ndarray, np.ndarray]:
    y = _load_csv(path, 'y')
    p = _load_csv(path, 'p')
    return y[:, idx], p[:, idx]


thresholds = np.arange(0, 1.0005, 0.0005)


def _get_all_scores_for(y: np.ndarray, p: np.ndarray, get_score: list[Callable[[np.ndarray, np.ndarray], float]]) -> list[list[float]]:
    p = scale_min_max(p)
    result = []
    for t in thresholds:
        row = [t] + [f(y, get_predictions(p, t)) for f in get_score]
        result.append(row)
    return result


def _print_all_scores_for(y: np.ndarray, p: np.ndarray) -> None:
    metric_names = ['accuracy', 'f1', 'informedness', 'markedness',
                    'mcc', 'precision', 'recall', 'fpr', 'fnr', 'vds']
    get_scores = [get_accuracy, get_f1, get_informedness, get_markedness,
                  get_mcc, get_precision, get_recall, get_fpr, get_fnr, get_vds]
    scores = _get_all_scores_for(y, p, get_scores)
    w = csv.writer(sys.stdout)
    w.writerow(['threshold']+metric_names)
    w.writerows(scores)


def _print_roc_for(y: np.ndarray, p: np.ndarray) -> None:
    metric_names = ['fpr', 'tpr']
    get_scores = [get_fpr, get_tpr]
    scores = _get_all_scores_for(y, p, get_scores)
    w = csv.writer(sys.stdout)
    w.writerow(['threshold']+metric_names)
    w.writerows(scores)


def _print_all_scores(path: str) -> None:
    y, p, _ = _get_results(path)
    _print_all_scores_for(y, p)


def _print_roc(path: str) -> None:
    y, p, _ = _get_results(path)
    _print_roc_for(y, p)


def _print_cwe_scores(path: str, idx: int) -> None:
    y, p = _get_cwe_results(path, idx)
    _print_all_scores_for(y, p)


def _get_threshold_scores_for(y: np.ndarray, p: np.ndarray, threshold: float, get_score: list[Callable[[np.ndarray, np.ndarray], float]]) -> list[float]:
    p = scale_min_max(p)
    return [f(y, get_predictions(p, threshold)) for f in get_score]


def _print_threshold_scores_for(threshold: float, row_name: str, y: np.ndarray, p: np.ndarray) -> None:
    get_scores = [get_accuracy, get_f1, get_informedness, get_markedness,
                  get_mcc, get_precision, get_recall, get_fpr, get_fnr, get_vds]
    scores = _get_threshold_scores_for(y, p, threshold, get_scores)
    w = csv.writer(sys.stdout)
    w.writerow([row_name]+scores)


def _print_threshold_scores(path: str, threshold: float, row_name: str) -> None:
    y, p, _ = _get_results(path)
    _print_threshold_scores_for(threshold, row_name, y, p)


def _print_threshold_cwe_scores(path: str, threshold: float, row_name: str, idx: int) -> None:
    y, p, _ = _get_cwe_results(path, idx)
    _print_threshold_scores_for(threshold, row_name, y, p)


def _print_threshold_scores_per_cwe(path: str, threshold: float, cwes: list[int]) -> None:
    metric_names = ['accuracy', 'f1', 'informedness', 'markedness',
                    'mcc', 'precision', 'recall', 'fpr', 'fnr', 'vds']
    y = _load_csv(path, 'y')
    p = _load_csv(path, 'p')
    w = csv.writer(sys.stdout)
    w.writerow(['CWE']+metric_names)
    for idx in range(1, len(cwes)+1):
        _print_threshold_scores_for(threshold, "CWE-"+str(cwes[idx-1]), y[:, idx], p[:, idx])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--depth", type=str, default=None)
    parser.add_argument("--cwe", type=int, default=None)
    parser.add_argument("--roc", type=bool, default=False)
    parser.add_argument("path", type=str, nargs=1)
    args = parser.parse_args()

    cwes = [20, 22, 78, 89, 94, 269, 306, 352, 400, 416, 434, 502, 787, 798, 918]

    if args.depth and args.threshold and args.cwe:
        _print_threshold_cwe_scores(args.path[0], args.threshold, args.depth, cwes.index(args.cwe))
    elif args.depth and args.threshold:
        _print_threshold_scores(args.path[0], args.threshold, args.depth)
    elif args.threshold:
        _print_threshold_scores_per_cwe(args.path[0], args.threshold, cwes)
    elif args.cwe:
        _print_cwe_scores(args.path[0], cwes.index(args.cwe))
    elif args.roc:
        _print_roc(args.path[0])
    else:
        _print_all_scores(args.path[0])
