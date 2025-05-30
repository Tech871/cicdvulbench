import os

from accelerate.utils import merge_fsdp_weights

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
from transformers import get_linear_schedule_with_warmup
import torch

from storage import get_dir, get_last_checkpoint, get_best_checkpoint


def get_model_path(classifier, s3=None):
    model_dir = classifier['pretrained_model_path']

    if s3 and model_dir.startswith('/'):
        return s3.get_cache_dir(model_dir)

    return model_dir


def get_fsdp_config(classifier):
    if 'qwen2' in classifier:
        layer = 'Qwen2DecoderLayer'
    elif 'qwen3' in classifier:
        layer = 'Qwen3DecoderLayer'
    else:
        return {}
    return {
        'transformer_layer_cls_to_wrap': [layer],
        'activation_checkpointing': True,
        'auto_wrap_policy': 'TRANSFORMER_BASED_WRAP',
        'backward_prefetch': 'BACKWARD_POST',
        'cpu_ram_efficient_loading': True,
        'forward_prefetch': False,
        'offload_params': True,
        'sharding_strategy': 'FULL_SHARD',
        'state_dict_type': 'SHARDED_STATE_DICT',
        'sync_module_states': True,
        'use_orig_params': True
    }


def load_pretrained_tokenizer(classifier, s3=None):
    model_path = get_model_path(classifier, s3)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return tokenizer


def load_pretrained_config(classifier, target, tokenizer, s3=None):
    model_path = get_model_path(classifier, s3)

    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = classifier.get('num_labels') or (
        2 if isinstance(target, int) else len(target) + 1
    )

    if 'problem_type' in classifier:
        config.problem_type = classifier['problem_type']

    config.pad_token_id = tokenizer.eos_token_id
    return config


def get_optimizers(model, train_dataset, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    num_training_steps = len(train_dataset) // args.per_device_train_batch_size
    num_training_steps *= args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def load_pretrained_model(classifier, target, args=None, s3=None):
    tokenizer = load_pretrained_tokenizer(classifier, s3)
    config = load_pretrained_config(classifier, target, tokenizer, s3)
    model_path = get_model_path(classifier, s3)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        device_map={"": args.device} if args else 'auto',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    # need to disable explicitly to enable gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    return model, tokenizer


def load_model(output_dir, config, args=None, s3=None):
    if s3:
        best_checkpoint = s3.get_last_checkpoint(output_dir)
        best_checkpoint = s3.get_cache_dir(best_checkpoint)
    else:
        last_checkpoint = get_last_checkpoint(output_dir)
        best_checkpoint = get_best_checkpoint(last_checkpoint)

    model_dir = f'{best_checkpoint}/pytorch_model_fsdp_0'
    if os.path.exists(model_dir):
        merge_fsdp_weights(model_dir, best_checkpoint)

    model = AutoModelForSequenceClassification.from_pretrained(
        best_checkpoint,
        device_map={"": args.device} if args else 'auto',
        config=config
    )
    if s3:
        s3.clean_cache()

    model.eval()
    return model


def get_output_dir(language, target, source, classifier):
    output_dir = get_dir(language, target, where='out', source=source, classifier=classifier)
    print('output_dir:', output_dir)
    return output_dir


def get_logging_dir(language, target, source, classifier):
    logging_dir = get_dir(language, target, where='log', source=source, classifier=classifier)
    print('logging_dir:', logging_dir)
    return logging_dir


def load_saved_model(classifier, language, target, train, args=None, s3=None):
    output_dir = get_output_dir(language, target, train, classifier['classifier'])

    tokenizer = load_pretrained_tokenizer(classifier, s3)
    config = load_pretrained_config(classifier, target, tokenizer, s3)
    model = load_model(output_dir, config, args, s3)

    return model, tokenizer


class OptimizedTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps):
        num_training_steps = len(self.train_dataset) // self.args.per_device_train_batch_size
        num_training_steps *= self.args.num_train_epochs

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps
        )
