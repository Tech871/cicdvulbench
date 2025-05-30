from catboost import CatBoostClassifier, CatBoostRegressor
from dataset import load_dataset
from storage import S3
from model import save_model, print_important_words, get_output_dir


def get_model(classifier, target, params):
    num_labels = classifier.get('num_labels') or (2 if isinstance(target, int) else len(target) + 1)

    if classifier.get('problem_type') == 'multi_label_classification':
        model = CatBoostRegressor(
            loss_function='MultiRMSE',
            **params
        )
    elif num_labels > 2:
        model = CatBoostClassifier(
            loss_function='MultiClass',
            classes_count=num_labels,
            custom_metric=['AUC'],
            **params
        )
    else:
        model = CatBoostClassifier(
            loss_function='Logloss',
            custom_metric=['AUC'],
            **params
        )

    return model


def main(classifier, language, target, train, s3=None):
    output_dir = get_output_dir(language, target, train, classifier['classifier'])
    print('log:', output_dir)

    x, y, vectorizer = load_dataset(classifier, language, target, train, s3=s3)

    params = {
        'depth': 8,
        'learning_rate': 0.1,
        'iterations': 100,
        'train_dir': output_dir,
        'random_seed': 42,
        'verbose': False
    }

    model = get_model(classifier, target, params)
    model.fit(x, y)

    print_important_words(x, vectorizer)

    save_model(model, vectorizer, output_dir, s3)
    return model, vectorizer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--classifier", type=str, default='tf-idf')
    parser.add_argument("--problem_type", type=str, default='multi_label_classification')
    parser.add_argument("--num_labels", type=int)

    parser.add_argument("--language", type=str, default='java')
    parser.add_argument("--target", type=str, default='20,502')

    parser.add_argument("--train", type=str, default='crawler')

    parser.add_argument("--s3_bucket", type=str)

    args = parser.parse_args()

    main(
        {
            'classifier': args.classifier,
            'problem_type': args.problem_type,
            'num_labels': args.num_labels
        },
        args.language,
        list(map(int, args.target.split(','))) if ',' in args.target else int(args.target),
        args.train,
        args.s3_bucket and S3(args.s3_bucket)
    )
