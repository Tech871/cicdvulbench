from storage import save_json, load_json
from target import get_targets


def main(language, _, year, license):
    vuls = []
    for pull in load_json('pulls', language, year, license):
        about = str(pull['title']) + ' ' + str(pull['body'])
        targets = get_targets(about)

        if len(targets):
            vuls.append({
                'targets': targets,
                'pull': pull['pull'],
                'repo': pull['repo']
            })
    save_json('vuls', vuls, language, year, license)


if __name__ == '__main__':
    main('java', None, 2025, 'mit')
