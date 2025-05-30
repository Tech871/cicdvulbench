from optuna import create_study

from train import get_model
from dataset import load_dataset
from sklearn.metrics import roc_auc_score
from model import get_output_dir

from metric.utils import get_is_pos, get_pos_probability


def get_p(classifier, model, x):
    if classifier.get('problem_type') == 'multi_label_classification':
        p = model.predict(x)
    else:
        p = model.predict_proba(x)
    return get_pos_probability(p)


def main(classifier, language, target, train, eval):
    output_dir = get_output_dir(language, target, eval, classifier['classifier'])
    print('log:', output_dir)

    x_train, y_train, vectorizer = load_dataset(classifier, language, target, train)
    x_eval, y_eval = load_dataset(classifier, language, target, eval, vectorizer)

    y = get_is_pos(y_eval)

    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', 6, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'iterations': 100,
            'train_dir': output_dir,
            'random_seed': 42,
            'verbose': False
        }
        model = get_model(classifier, target, params)
        model.fit(x_train, y_train)

        p = get_p(classifier, model, x_eval)
        return roc_auc_score(y, p)

    study = create_study(direction='maximize', study_name=classifier['classifier'])
    study.optimize(objective, n_trials=10)

    print('best trial:', study.best_trial)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--classifier", type=str, default='tf-idf')
    parser.add_argument("--problem_type", type=str, default='single_label_classification')
    parser.add_argument("--num_labels", type=int)

    parser.add_argument("--language", type=str, default='java')
    parser.add_argument("--target", type=str, default='20,502')

    parser.add_argument("--train", type=str, default='crawler')
    parser.add_argument("--eval", type=str, default='osv_dev')

    args = parser.parse_args()

    main(
        {
            'classifier': args.classifier,
            'problem_type': args.problem_type,
            'num_labels': args.num_labels
        },
        args.language,
        list(map(int, args.target.split(','))) if ',' in args.target else int(args.target),
        {
            'train': args.train,
            'eval': args.eval
        }
    )
