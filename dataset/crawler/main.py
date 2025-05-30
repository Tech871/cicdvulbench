from github import repos, pulls, commits, files, rows
import json

import vuls

from threading import Thread, Semaphore
from github import get_tokens

tokens = get_tokens()
token_semaphore = Semaphore(len(tokens))


def task(language, year, license):
    with token_semaphore:
        token = tokens.pop()
        for what in [repos, pulls, vuls, commits, files]:
            print(what.__name__, language, year, license)
            what.main(language, token, year, license)
        tokens.append(token)


def main(languages, years, licenses):
    threads = []
    for language in languages:
        for year in years:
            for license in licenses:
                threads.append(
                    Thread(target=task, args=(language, year, license))
                )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    rows.main(languages, 'crawler', years, licenses)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.json')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as src:
        config = json.load(src)

    args = parser.parse_args()

    main(
        config['languages'],
        config['years'],
        config['licenses']
    )
