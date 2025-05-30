from github import rows

import files
from storage import get_s3


def main(languages, source, s3=None):
    for language in languages:
        files.main(language, source, s3=s3)

    rows.main(languages, source, s3=s3)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", type=str, default='java')
    parser.add_argument("--source", type=str, default='qwen_classified_depth_0')
    parser.add_argument("--s3_bucket", type=str)

    args = parser.parse_args()

    main(args.languages.split(','), args.source, get_s3(args))
