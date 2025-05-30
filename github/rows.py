import pandas as pd
from collections import defaultdict

from storage import save_csv, load_json, get_path


def read(rows, language, year, license, s3=None):
    vuls = load_json('files', language, year, license, s3)

    for vul in vuls:
        before = vul.get('before')
        after = vul.get('after')

        for target in vul['targets']:
            if before:
                rows[target].append({
                    'target': 1,
                    'code': before
                })
            if after:
                rows[target].append({
                    'target': 0,
                    'code': after
                })


def save(source, language, rows, s3=None):
    for target, dataset in rows.items():
        dataset = pd.DataFrame(dataset, columns=['target', 'code'])

        save_csv('code', dataset['code'], language, target, source=source, s3=s3)
        save_csv('target', dataset['target'], language, target, source=source, s3=s3)


def log(source, language, rows):
    log_dir = get_path('log', language)
    with open(f'{log_dir}/{source}.txt', 'w') as file:
        for target in sorted(rows.keys()):
            print(target, '\t', len(rows[target]), file=file)


def main(languages, source, years=(None,), licenses=(None,), s3=None):
    for language in languages:
        rows = defaultdict(list)

        for year in years:
            for license in licenses:
                read(rows, language, year, license, s3)

        save(source, language, rows, s3)
        log(source, language, rows)


if __name__ == '__main__':
    main(['java'], 'maven')
