import os
from storage import get_path


def is_400(about):
    return 'DoS' in about


def load_rules(key):
    rules = {}
    path = get_path('target', 'cwe', key)
    for cwe in os.listdir(path):
        with open(f'{path}/{cwe}') as src:
            for substr in src:
                rules[substr.strip()] = int(cwe)
    return rules


includes = load_rules('includes')
excludes = load_rules('excludes')


def get_targets(about):
    if is_400(about):
        return [400]

    about = about.lower()

    excluded = set()
    for substr, cwe in excludes.items():
        if substr in about:
            excluded.add(cwe)

    included = set()
    for substr, cwe in includes.items():
        if cwe not in excluded and substr in about:
            included.add(cwe)

    return list(included)


def rename_cwe():
    for key in ['includes', 'excludes']:
        path = get_path('target', 'cwe', key)
        for file in os.listdir(path):
            os.rename(f'{path}/{file}', f'{path}/{file[3:-4]}')
