from client import get_prediction
import pandas as pd
from storage import save_csv
from dataset import load_xy


def main(classifier, language, target, test):
    def to_csv(what, values):
        save_csv(
            what,
            pd.DataFrame(values),
            language,
            target,
            test=test,
            classifier=classifier['classifier']
        )

    x, y = load_xy(language, target, test, classifier)
    to_csv('y', y)

    p = [get_prediction(classifier, language, target, code) for code in x]
    to_csv('p', p)
