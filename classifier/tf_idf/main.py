import json

from train import main as train_classifier
from test import main as test_classifier

from storage import S3

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.json')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as src:
        config = json.load(src)

    args = parser.parse_args()

    bucket = config.get('s3_bucket')
    s3 = bucket and S3(bucket)

    for language in config['languages']:
        for target in config['targets']:
            for classifier in config['classifiers']:
                for train in config['trains']:
                    model, vectorizer = train_classifier(classifier, language, target, train, s3)

                    for test in config['tests']:
                        test_classifier(
                            classifier, language, target, train, test,
                            s3, model=model, vectorizer=vectorizer,
                        )
