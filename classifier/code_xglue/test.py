import pandas as pd

from model import load_saved_model, get_output_dir
from args import TrainingArguments
from dataset import load_dataset
from trainer import Trainer
from storage import save_csv
from utils import set_seed


def main(classifier, language, target, train, test, model=None, tokenizer=None):
    def to_csv(what, values):
        save_csv(
            what,
            pd.DataFrame(values),
            language,
            target,
            train=train,
            test=test,
            classifier=classifier['classifier']
        )

    output_dir = get_output_dir(language, target, test, classifier['classifier'])
    print('log:', output_dir)

    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4
    )

    if model is None or tokenizer is None:
        model, tokenizer = load_saved_model(classifier, target, args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args
    )

    dataset = load_dataset(language, target, test, tokenizer, classifier)
    to_csv('y', dataset.labels)

    p = trainer.predict(dataset)
    to_csv('p', p)


if __name__ == '__main__':
    set_seed()

    main(
        {
            'classifier': 'bert',
            'pretrained_model_path': 'bert-base-uncased',
            'problem_type': 'single_label_classification',
            'num_labels': 2
        },
        'java',
        [
            20,
            502
        ],
        'qwen_classified_depth_0',
        'osv_dev_depth_0'
    )
