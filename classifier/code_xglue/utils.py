import os
import numpy as np

import torch
import random

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from huggingface_hub import login
from metric.utils import get_is_pos, get_pos_probability

from sklearn.metrics import roc_auc_score


from dotenv import load_dotenv

load_dotenv()


def login_huggingface():
    login(
        token=os.getenv('HUGGINGFACE_TOKEN'),
        add_to_git_credential=True
    )


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_average(logits, labels):
    if len(labels.shape) > 1:
        return 'macro'
    if logits.shape[1] > 2:
        return 'micro'
    return 'binary'


def get_predictions(labels, logits):
    if len(logits.shape) == 1:
        return logits

    i = np.argmax(logits, axis=1)
    if len(labels.shape) == 1:
        return i

    predictions = np.zeros_like(logits)
    j = np.arange(i.size)
    predictions[j, i] = 1

    return predictions


def compute_metrics(result):
    logits, labels = result

    predictions = get_predictions(labels, logits)
    average = get_average(labels, logits)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=np.nan
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": fscore
    }


def compute_binary_metrics(result):
    logits, labels = result

    p = get_pos_probability(logits)
    y = get_is_pos(labels)
    roc_auc = roc_auc_score(y, p)

    accuracy = accuracy_score(y, p > 0.5)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y, p > 0.5, average='binary', zero_division=np.nan
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "f1": fscore
    }
