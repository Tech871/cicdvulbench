from storage import save_json, load_json, has_json
from progress import save_progress

from github import get_pulls, get_token


def is_ok(repo):
    return repo['size'] >= 10 and repo['watchers'] >= 1


def main(language, token, year, license):
    if has_json('pulls', language, year, license):
        return

    repos = load_json('repos', language, year, license)
    repos = list(filter(is_ok, repos))

    progress = save_progress('pulls', len(repos), language, year, license)

    pulls = []
    for i, repo in enumerate(repos, 1):
        pulls += get_pulls(repo, token)
        progress(i)

    save_json('pulls', pulls, language, year, license)


if __name__ == '__main__':
    main('java', get_token(), 2025, 'mit')
