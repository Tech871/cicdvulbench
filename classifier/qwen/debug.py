from transformers import Trainer, TrainingArguments
from dataset import load_dataset
from utils import login_huggingface, set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch

pretrained_model_path = 'Qwen/Qwen2.5-Coder-0.5B'


def load_pretrained_tokenizer(cache_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path,
        use_fast=True,
        cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return tokenizer


def load_pretrained_config(tokenizer):
    config = AutoConfig.from_pretrained(pretrained_model_path)
    config.num_labels = 2

    config.pad_token_id = tokenizer.eos_token_id
    return config


def get_pretrained_model(config, args):
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_path,
        config=config,
        device_map={"": args.device},
        torch_dtype=torch.bfloat16,
    )


def main():
    args = TrainingArguments(
        save_strategy='epoch',
        eval_strategy='epoch',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        gradient_accumulation_steps=5,
        weight_decay=0.01,
        learning_rate=5e-5,
        push_to_hub=False,
        ddp_find_unused_parameters=False
    )
    tokenizer = load_pretrained_tokenizer()
    config = load_pretrained_config(tokenizer)
    model = get_pretrained_model(config, args)

    train_dataset = load_dataset('java', 20, 'crawler', tokenizer, {})

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        processing_class=tokenizer
    )
    trainer.train()


if __name__ == '__main__':
    set_seed()
    login_huggingface()

    main()
