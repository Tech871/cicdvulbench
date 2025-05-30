import re
import json

import pandas as pd

from storage import save_csv, load_csv, S3


def code2words(code):
    return ' '.join(word for word in re.split(r'\W+', str(code).lower()) if word)


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
            for source in config['sources']:
                code = load_csv('code', language, target, source=source, s3=s3)
                words = pd.Series(map(code2words, code))
                save_csv('words', words, language, target, source=source, s3=s3)
