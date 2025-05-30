import pandas as pd

from transformers import Trainer, TrainingArguments
from dataset import load_dataset
from model import load_saved_model, get_logging_dir
from storage import save_csv, get_s3
from utils import login_huggingface, set_seed
from utils import get_classifier, get_target

import torch


def main(classifier, language, target, train, test, s3=None, trainer=None, tokenizer=None):
    def to_csv(what, values):
        save_csv(
            what,
            pd.DataFrame(values),
            language,
            target,
            train=train,
            test=test,
            classifier=classifier['classifier'],
            s3=s3
        )

    if trainer is None:
        logging_dir = get_logging_dir(language, target, test, classifier['classifier'])

        args = TrainingArguments(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            logging_dir=logging_dir,
            push_to_hub=False
        )
        model, tokenizer = load_saved_model(classifier, language, target, train, s3=s3)

        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            args=args
        )

    dataset = load_dataset(language, target, test, tokenizer, classifier, s3=s3)
    p = trainer.predict(dataset).predictions

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if trainer.args.local_rank > 0:
        return

    to_csv('y', dataset['y'])
    to_csv('p', p)


if __name__ == '__main__':
    set_seed()
    login_huggingface()

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--classifier", type=str)
    parser.add_argument("--pretrained_model_path", type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument("--problem_type", type=str, default='multi_label_classification')
    parser.add_argument("--num_labels", type=int)

    parser.add_argument("--language", type=str, default='java')
    parser.add_argument("--target", type=str, default='20,502')

    parser.add_argument("--train", type=str, default='qwen_classified_depth_0')
    parser.add_argument("--test", type=str, default='osv_dev_depth_0')

    parser.add_argument("--s3_bucket", type=str)

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
        args.train,
        args.test,
        get_s3(args)
    )
