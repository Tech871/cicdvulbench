from storage import load_jsonl, save_json
from target import get_targets


def take(dst):
    return {
        'before': dst['code_before'],
        'after': dst['code_after'],
        'targets': get_targets(' '.join(dst['cwe']).replace('_', '-'))
    }


def main(language, source, s3=None):
    vuls = [take(vul) for vul in load_jsonl(language, source, s3)]
    save_json('files', vuls, language, s3=s3)


if __name__ == '__main__':
    main('java', 'osv_dev')
