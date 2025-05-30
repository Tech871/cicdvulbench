import pandas as pd

from storage import save_csv, S3
from model import load_saved_model
from dataset import load_dataset
from typing import Any
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np


def get_p(classifier, model, x):
    if classifier.get('problem_type') == 'multi_label_classification':
        p = model.predict(x)
    else:
        p = model.predict_proba(x)
    return p


def main(classifier, language, target, train, test, s3=None, model=None, vectorizer=None):
    def to_csv(what, values):
        save_csv(
            what,
            pd.DataFrame(values),
            language,
            target,
            train=train,
            test=test,
            classifier=classifier['classifier'],
            s3=s3
        )

    if model is None or vectorizer is None:
        model, vectorizer = load_saved_model(classifier, language, target, train, s3)

    x, y = load_dataset(classifier, language, target, test, vectorizer, s3)
    to_csv('y', y)

    p = get_p(classifier, model, x)
    to_csv('p', p)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--classifier", type=str, default='tf-idf')
    parser.add_argument("--problem_type", type=str, default='multi_label_classification')
    parser.add_argument("--num_labels", type=int)

    parser.add_argument("--language", type=str, default='java')
    parser.add_argument("--target", type=str, default='20,502')

    parser.add_argument("--train", type=str, default='crawler')
    parser.add_argument("--test", type=str, default='osv_dev')

    parser.add_argument("--s3_bucket", type=str)

    args = parser.parse_args()

    main(
        {
            'classifier': args.classifier,
            'problem_type': args.problem_type,
            'num_labels': args.num_labels
        },
        args.language,
        list(map(int, args.target.split(','))) if ',' in args.target else int(args.target),
        args.train,
        args.test,
        args.s3_bucket and S3(args.s3_bucket)
    )
