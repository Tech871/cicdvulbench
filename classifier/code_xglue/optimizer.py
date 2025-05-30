import os
import torch
from transformers import get_linear_schedule_with_warmup


def get_optimizer(model, args):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0
        }
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    return optimizer


def get_scheduler(optimizer, dataset, args):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_train_epochs * len(dataset) // 10,
        num_training_steps=len(dataset)
    )


def load_optimizer(args, optimizer):
    src = f'{args.output_dir}/optimizer.pt'
    if os.path.exists(src):
        optimizer.load_state_dict(torch.load(src))


def load_scheduler(args, scheduler):
    src = f'{args.output_dir}/scheduler.pt'
    if os.path.exists(src):
        scheduler.load_state_dict(torch.load(src))
