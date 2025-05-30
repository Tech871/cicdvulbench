import json
import subprocess

import pandas as pd
from time import sleep

from dataset import load_xy
from storage import save_csv, save_code, S3
import re


def get_target(cwe):
    match = re.search(r'^CWE-(\d+)', cwe)
    return match and int(match.group(1))


def get_num_labels(target, classifier):
    num_labels = classifier.get('num_labels')

    if isinstance(target, int):
        if num_labels != 1:
            return 2
    elif isinstance(target, list):
        if num_labels not in (1, 2):
            return len(target) + 1
    else:
        raise TypeError(type(target))

    return num_labels


def get_p(target, targets, num_labels):
    neg = int(len(targets) == 0)
    q = 1 - neg

    if isinstance(target, int):
        pos = int(target in targets)
        pos = (pos + q) / 2

    elif isinstance(target, list):
        pos = [int(t in targets) for t in target]
        n = len(target)
        if num_labels > 2:
            pos = [(n * p + q) / (n + 1) for p in pos]
            return [neg] + [round(p, 2) for p in pos]
        pos = (sum(pos) + q) / (n + 1)
    else:
        raise TypeError(type(target))

    pos = round(pos, 2)
    return pos if num_labels == 1 else [neg, pos]


def get_prediction(language, target, code, num_labels, config, s3=None):
    if not code.strip():
        return get_p(target, set(), num_labels)

    file_path = save_code(code, language, s3)

    try:
        sleep(1)
        cmd = ['semgrep', f'--config={config}', '--json', file_path]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)

        results = json.loads(res.stdout).get('results', [])
        targets = set()

        for result in results:
            metadata = result.get('extra', {}).get('metadata', {})
            for cwe in metadata.get('cwe', []):
                targets.add(get_target(cwe))

        return get_p(target, targets, num_labels)

    except subprocess.CalledProcessError as e:
        print(e.stderr)
    except FileNotFoundError as e:
        print(e)

    return 0


def main(classifier, language, target, test, s3=None):
    def to_csv(what, values):
        save_csv(
            what,
            pd.DataFrame(values),
            language,
            target,
            test=test,
            classifier=classifier['classifier'],
            s3=s3,
        )

    x, y = load_xy(language, target, test, classifier, s3)
    to_csv('y', y)

    num_labels = get_num_labels(target, classifier)
    config = {language: classifier['rule']}
    p = [get_prediction(language, target, code, num_labels, config, s3) for code in x]
    to_csv('p', p)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--classifier", type=str, default='semgrep')
    parser.add_argument("--rule", type=str, default='p/java')
    parser.add_argument("--problem_type", type=str, default='single_label_classification')
    parser.add_argument("--num_labels", type=int)

    parser.add_argument("--language", type=str, default='java')
    parser.add_argument("--target", type=str, default='20,502')

    parser.add_argument("--test", type=str, default='osv_dev')

    parser.add_argument("--s3_bucket", type=str)

    args = parser.parse_args()

    main(
        {
            'classifier': args.classifier,
            'rule': args.rule,
            'problem_type': args.problem_type,
            'num_labels': args.num_labels
        },
        args.language,
        list(map(int, args.target.split(','))) if ',' in args.target else int(args.target),
        args.test,
        args.s3_bucket and S3(args.s3_bucket)
    )
