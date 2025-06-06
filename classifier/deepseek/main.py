import json

from test import main as test_classifier

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.json')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as src:
        config = json.load(src)

    for language in config['languages']:
        for target in config['targets']:
            for classifier in config['classifiers']:
                for test in config['tests']:
                    test_classifier(classifier, language, target, test)
