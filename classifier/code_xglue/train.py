from trainer import Trainer
from args import TrainingArguments

from model import load_pretrained_model, get_output_dir
from dataset import load_datasets
from optimizer import get_optimizer, get_scheduler

from utils import compute_binary_metrics, set_seed


def main(classifier, language, target, train, eval):
    output_dir = get_output_dir(language, target, train, classifier['classifier'])
    print('log:', output_dir)

    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        learning_rate=5e-5,
        adam_epsilon=1e-8,
        metric_for_best_model="roc_auc",
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
        gradient_accumulation_steps=5,
        max_grad_norm=1,
        local_rank=-1,
        no_cuda=True,
        fp16=False,
        opt_level="O1"
    )

    model, tokenizer = load_pretrained_model(classifier, target, args)
    train_dataset, eval_dataset = load_datasets(language, target, train, eval, tokenizer, classifier)

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, train_dataset, args)

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_binary_metrics
    )
    trainer.train()

    return model, tokenizer


if __name__ == '__main__':
    set_seed()

    main(
        {
            'classifier': 'roberta',
            'pretrained_model_path': 'roberta-base',
            'problem_type': 'single_label_classification',
            'num_labels': 2,
            'layout': 'ReGGNN'
        },
        'java',
        [
            20,
            502
        ],
        'qwen_classified_depth_0',
        'osv_dev_depth_0'
    )
