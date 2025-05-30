import os

import json
from storage import get_path, save_json
from github import get_repo, get_sha, get_pull
from target import get_targets


def main(language, token=None):
    vuls = []

    path = get_path('src', language, 'maven')
    for name in os.listdir(path):
        with open(f'{path}/{name}', encoding="utf-8") as src:
            data = json.load(src)

            aliases = data.get('aliases', [])
            if 'database_specific' in data:
                aliases += data['database_specific']['cwe_ids']

            about = ' '.join(aliases)
            targets = get_targets(about)
            if not len(targets):
                continue

            shas = set()
            pulls = set()

            vul = {'targets': targets}

            for ref in data.get('references', []):
                if ref['type'] != 'WEB':
                    continue
                url = ref['url']

                repo = get_repo(url)
                if not repo:
                    continue

                sha = get_sha(url)
                pull = get_pull(url)
                if not (sha or pull):
                    continue

                if 'repo' not in vul:
                    vul['repo'] = repo
                if repo != vul['repo']:
                    continue

                if sha:
                    shas.add(sha)
                if pull:
                    pulls.add(pull)

            if len(shas):
                for sha in shas:
                    vul['commits'] = [{'sha': sha}]
                    vuls.append(vul.copy())
            else:
                for pull in pulls:
                    vul['pull'] = pull
                    vuls.append(vul.copy())

    save_json('vuls', vuls, language)


if __name__ == '__main__':
    main('java')
