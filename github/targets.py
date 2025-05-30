from target import get_targets
from storage import load_json, save_json


def is_ok(vul):
    commits = vul.get('commits', [])
    pulls = vul.get('pulls', [])
    return len(vul['targets']) and (len(commits) or len(pulls))


def main(language, year=None, license=None):
    vuls = load_json('vuls', language, year, license)

    for i, vul in enumerate(vuls):
        vul['targets'] = get_targets(vul['text'])

    vuls = list(filter(is_ok, vuls))
    save_json('targets', vuls, language, year, license)


if __name__ == '__main__':
    main('java')
