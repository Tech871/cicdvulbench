import numpy as np
from optuna import create_study

from sklearn.metrics import roc_auc_score
from model import get_output_dir

from metric.utils import get_is_pos, get_pos_probability
from utils import compute_binary_metrics, login_huggingface, set_seed
from utils import get_classifier, get_target

from transformers import TrainingArguments, EarlyStoppingCallback
from model import load_pretrained_model, load_pretrained_tokenizer
from model import OptimizedTrainer
from dataset import load_datasets


def main(classifier, language, target, train):
    output_dir = get_output_dir(language, target, train['eval'], classifier['classifier'])
    tokenizer = load_pretrained_tokenizer(classifier)
    train_dataset, eval_dataset = load_datasets(language, target, train, tokenizer, classifier)

    def objective(trial):
        args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            save_strategy='epoch',
            eval_strategy='epoch',
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            gradient_accumulation_steps=trial.suggest_int('gradient_accumulation_steps', 5, 10),
            weight_decay=trial.suggest_float('weight_decay', 0.001, 0.1),
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2),
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model='roc_auc',
            push_to_hub=False,
            ddp_find_unused_parameters=False,
            bf16=True
        )

        model, tokenizer = load_pretrained_model(classifier, target)

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
        callbacks = [early_stopping_callback]

        trainer = OptimizedTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_binary_metrics,
            callbacks=callbacks
        )
        trainer.train()

        p = get_pos_probability(trainer.predict(eval_dataset).predictions)
        y = get_is_pos(np.array(eval_dataset['y']))

        return roc_auc_score(y, p)

    study = create_study(direction='maximize', study_name=classifier['classifier'])
    study.optimize(objective, n_trials=10)

    print('best trial:', study.best_trial)


if __name__ == '__main__':
    set_seed()
    login_huggingface()

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--classifier", type=str)
    parser.add_argument("--pretrained_model_path", type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument("--problem_type", type=str, default='single_label_classification')
    parser.add_argument("--num_labels", type=int)

    parser.add_argument("--language", type=str, default='java')
    parser.add_argument("--target", type=str, default='20,502')

    parser.add_argument("--train", type=str, default='qwen_classified_depth_0')
    parser.add_argument("--eval", type=str, default='osv_dev_depth_0')

    args = parser.parse_args()

    main(
        {
            'classifier': get_classifier(args),
            'pretrained_model_path': args.pretrained_model_path,
            'problem_type': args.problem_type,
            'num_labels': args.num_labels
        },
        args.language,
        get_target(args),
        {
            'train': args.train,
            'eval': args.eval
        }
    )
