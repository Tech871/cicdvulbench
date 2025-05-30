import accuracy
import f1_score
import informedness
import markedness
import mcc
import pr
import roc
import pass_at_k
from storage import get_dir, to_str, S3

from utils import get_results

import json

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.json')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as src:
        config = json.load(src)

    bucket = config.get('s3_bucket')
    s3 = bucket and S3(bucket)

    for language in config['languages']:
        for target in config['targets']:
            for test in config['tests']:
                save_dir = get_dir(language, target, test=test)

                title = f'{language} {test}\nCWE-{to_str(target)}'
                ks, ps = get_results(language, target, test, config['models'], s3, binary=True)

                for metric in [accuracy, f1_score, informedness, markedness, mcc, pr, roc, pass_at_k]:
                    metric.main(ks, ps, title, save_dir)
