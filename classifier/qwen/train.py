from transformers import EarlyStoppingCallback
from transformers import TrainingArguments

from dataset import load_datasets
from model import get_output_dir, get_logging_dir, get_fsdp_config
from model import load_pretrained_model, OptimizedTrainer
from storage import SaveToS3Callback, get_s3
from utils import compute_binary_metrics, login_huggingface, set_seed
from utils import get_classifier, get_target


def main(classifier, language, target, train, s3=None, checkpoint=None):
    output_dir = get_output_dir(language, target, train['train'], classifier['classifier'])
    logging_dir = get_logging_dir(language, target, train['train'], classifier['classifier'])

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        logging_dir=logging_dir,
        logging_steps=50,
        save_strategy='best',
        eval_strategy='steps',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        learning_rate=2e-5,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='roc_auc',
        push_to_hub=False,
        ddp_find_unused_parameters=False,
        bf16=True,
        fsdp='full_shard auto_wrap offload',
        fsdp_config=get_fsdp_config(classifier['classifier'])
    )

    model, tokenizer = load_pretrained_model(classifier, target, args, s3)

    train_dataset, eval_dataset = load_datasets(language, target, train, tokenizer, classifier, s3=s3)

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=4,
            early_stopping_threshold=0.001
        )
    ]
    if s3 and checkpoint != -1:
        callbacks.append(SaveToS3Callback(s3, args))

    trainer = OptimizedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_binary_metrics,
        callbacks=callbacks
    )
    if checkpoint != -1:
        checkpoint = f'{output_dir}/checkpoint-{checkpoint}'
        if s3:
            checkpoint = s3.get_cache_dir(checkpoint)
            trainer.train(resume_from_checkpoint=checkpoint)
            s3.clean_cache()
        else:
            trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    return trainer, tokenizer


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
    parser.add_argument("--eval", type=str)

    parser.add_argument("--s3_bucket", type=str)
    parser.add_argument("--checkpoint", type=int)

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
        },
        get_s3(args),
        args.checkpoint
    )
