from storage import save_json, load_json, has_json

from github import get_compared_files, get_file, get_previous_sha, get_token
from github import is_modified, get_change_score

from progress import save_progress


def take(file):
    return {
        'filename': file['filename'],
        'deletions': file['deletions'],
        'additions': file['additions']
    }


def compare(language, vul, token):
    repo = vul['repo']
    after_sha = vul['commits'][-1]['sha']

    before_sha = get_previous_sha(repo, vul['commits'][0]['sha'], token)
    files = get_compared_files(repo, before_sha, after_sha, token)

    vul['files'] = list(map(take, filter(is_modified(language), files)))
    changes = [(get_change_score(file), file['filename']) for file in vul['files']]

    if not len(changes):
        return

    changes, filename = max(changes)
    vul['before'] = get_file(repo, filename, before_sha, token)
    vul['after'] = get_file(repo, filename, after_sha, token)


def main(language, token, year=None, license=None):
    if has_json('files', language, year, license):
        return

    vuls = load_json('commits', language, year, license)

    progress = save_progress('files', len(vuls), language, year, license)
    for i, vul in enumerate(vuls, 1):
        compare(language, vul, token)
        progress(i)

    save_json('files', vuls, language, year, license)


if __name__ == '__main__':
    main('java', get_token())
