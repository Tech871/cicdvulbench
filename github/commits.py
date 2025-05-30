from storage import load_json, save_json, has_json

from github import get_token, get_commits
from progress import save_progress


def is_ok(vul):
    return len(vul['commits'])


def main(language, token, year=None, license=None):
    if has_json('commits', language, year, license):
        return

    vuls = load_json('vuls', language, year, license)

    progress = save_progress('commits', len(vuls), language, year, license)

    for i, vul in enumerate(vuls, 1):
        if 'pull' in vul:
            vul['commits'] = get_commits(vul, token)
        progress(i)

    vuls = list(filter(is_ok, vuls))
    save_json('commits', vuls, language, year, license)


if __name__ == '__main__':
    main('java', get_token())
