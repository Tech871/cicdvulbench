import torch

from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from storage import load_csv

import cityhash
import numpy as np

from sklearn.model_selection import train_test_split


class CodeDataset(Dataset):
    def __init__(self, x, y, tokenizer=None):
        def get_tokens(code):
            tokens = tokenizer.tokenize(code)
            n = tokenizer.num_tokens
            tokens = [tokenizer.cls_token] + tokens[:n - 2] + [tokenizer.sep_token]
            tokens = tokenizer.convert_tokens_to_ids(tokens)
            return tokens + [tokenizer.pad_token_id] * (n - len(tokens))

        self.tokens = list(map(get_tokens, x)) if tokenizer else x
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.tensor(self.tokens[i]), torch.tensor(self.labels[i])

    def train_test_split(self, test_size, shuffle=True, seed=None):
        train_tokens, test_tokens, train_labels, test_labels = train_test_split(
            self.tokens, self.labels,
            test_size=test_size, shuffle=shuffle, random_state=seed
        )
        return {
            'train': CodeDataset(train_tokens, train_labels),
            'test': CodeDataset(test_tokens, test_labels),
        }


def single_to_multi(x, y):
    hashes = list(map(cityhash.CityHash64, x))
    hr, ir = np.unique(hashes, return_inverse=True)
    ri = np.zeros(len(hr), dtype=int)
    for i, r in enumerate(ir):
        ri[r] = i
    rx = x[ri]
    ry = np.zeros((len(hr), max(y) + 1), dtype=int)
    ry[ir, y] = 1
    return rx, torch.from_numpy(ry).to(torch.float)


def get_xy(language, target, source):
    x = load_csv('code', language, target, source=source)
    y = load_csv('target', language, target, source=source)
    return x, y


def concat_xy(language, targets, source, classifier):
    xs = []
    ys = []
    if len(targets) == 1 or classifier.get('num_labels'):
        for target in targets:
            x, y = get_xy(language, target, source)
            xs.append(x)
            ys.append(y)
    else:
        for label, target in enumerate(targets, 1):
            x, y = get_xy(language, target, source)
            xs.append(x)
            ys.append(label * y)

    y = np.concatenate(ys)
    x = np.concatenate(xs)
    return x, y


def load_xy(language, target, source, classifier):
    if isinstance(target, int):
        x, y = get_xy(language, target, source)
    elif isinstance(target, list):
        x, y = concat_xy(language, target, source, classifier)

        if classifier.get('problem_type') == 'multi_label_classification':
            x, y = single_to_multi(x, y)
    else:
        raise TypeError(type(target))

    return x, y


def load_dataset(language, target, source, tokenizer, classifier):
    x, y = load_xy(language, target, source, classifier)
    return CodeDataset(x, y, tokenizer)


def load_datasets(language, target, train, eval, tokenizer, classifier):
    train_dataset = load_dataset(language, target, train, tokenizer, classifier)

    if eval:
        eval_dataset = load_dataset(language, target, train, tokenizer, classifier)
        return train_dataset, eval_dataset

    dataset = train_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    return dataset['train'], dataset['test']


def get_dataloader(dataset, batch_size, local_rank=-1):
    Sampler = RandomSampler if local_rank == -1 else DistributedSampler
    return DataLoader(
        dataset,
        sampler=Sampler(dataset),
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
