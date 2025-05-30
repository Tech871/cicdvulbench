import json

from train import main as train_classifier
from test import main as test_classifier

from utils import login_huggingface, set_seed
from storage import S3

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.json')
    args = parser.parse_args()

    set_seed()
    login_huggingface()

    with open(args.config, encoding='utf-8') as src:
        config = json.load(src)

    bucket = config.get('s3_bucket')
    s3 = bucket and S3(bucket)

    for language in config['languages']:
        for target in config['targets']:
            for classifier in config['classifiers']:
                if 'classifier' not in classifier:
                    classifier['classifier'] = classifier['pretrained_model_path'].split('/')[-1].lower()

                for train in config['trains']:
                    if isinstance(train, str):
                        train = {
                            'train': train
                        }
                    trainer, tokenizer = train_classifier(classifier, language, target, train, s3, checkpoint=-1)

                    for test in config['tests']:
                        test_classifier(
                            classifier, language, target, train['train'], test,
                            s3, trainer=trainer, tokenizer=tokenizer,
                        )
