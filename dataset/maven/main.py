from github import commits, files, rows

import vuls

from threading import Thread, Semaphore
from github import get_tokens

tokens = get_tokens()
token_semaphore = Semaphore(len(tokens))


def task(language):
    with token_semaphore:
        token = tokens.pop()
        for what in [vuls, commits, files]:
            print(what.__name__, language)
            what.main(language, token)

        tokens.append(token)


def main(languages):
    threads = []
    for language in languages:
        threads.append(
            Thread(target=task, args=(language,))
        )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    rows.main(languages, 'maven')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", type=str, default='java')

    args = parser.parse_args()

    main(args.languages.split(','))
