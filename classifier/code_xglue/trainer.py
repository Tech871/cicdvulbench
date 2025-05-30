import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from dataset import get_dataloader
from model import save_model


class Trainer:

    def __init__(
            self,
            args,
            model,
            tokenizer,
            optimizer=None,
            scheduler=None,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=None
    ):
        self.output_dir = args.output_dir

        self.compute_metrics = compute_metrics
        self.metric_for_best_model = args.metric_for_best_model
        self.evaluation_strategy = args.evaluation_strategy

        self.num_train_epochs = args.num_train_epochs
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_grad_norm = args.max_grad_norm

        self.eval_batch_size = args.per_device_eval_batch_size * args.num_devices
        self.train_batch_size = args.per_device_train_batch_size * args.num_devices

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.local_rank = args.local_rank
        self.num_devices = args.num_devices
        self.device = args.device

        self.early_stopping_threshold = args.early_stopping_threshold
        self.early_stopping_patience = args.early_stopping_patience

        model.to(args.device)
        self.amp = args.amp
        if self.amp:
            self.model, self.optimizer = self.amp.initialize(model, optimizer, opt_level=args.opt_level)
        else:
            self.model = model
            self.optimizer = optimizer

        self.scheduler = scheduler
        self.tokenizer = tokenizer

    def train(self):
        if self.num_devices > 1:
            self.model = nn.DataParallel(self.model)

        if self.local_rank != -1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )

        early_stopping_counter = 0

        best = {
            'score': 0
        }

        dataloader = get_dataloader(self.train_dataset, self.train_batch_size, self.local_rank)

        for epoch in range(self.num_train_epochs):
            sum_loss = 0
            avg_loss = 0

            bar = tqdm(dataloader, total=len(dataloader))

            for step, batch in enumerate(bar, 1):
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                self.model.train()
                loss, logits = self.model(inputs, labels)

                if self.num_devices > 1:
                    loss = loss.mean()
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                if self.amp:
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.amp.master_params(self.optimizer),
                        self.max_grad_norm
                    )
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                sum_loss += loss.item()
                avg_loss = round(sum_loss / step, 5)

                bar.set_description(f'epoch {epoch}, loss {avg_loss}')

                if step % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

            if self.local_rank == -1 and self.evaluation_strategy == 'epoch':
                score = self.evaluate()

                if score > best['score']:
                    best['score'] = score
                    save_model(self.model, self.output_dir)

            if self.early_stopping_patience > 0:
                if 'loss' not in best or best['loss'] - avg_loss > self.early_stopping_threshold:
                    best['loss'] = avg_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= self.early_stopping_patience:
                        break

    def evaluate(self):
        self.model.eval()

        logits = []
        labels = []

        eval_dataloader = get_dataloader(self.eval_dataset, self.eval_batch_size, self.local_rank)

        for batch in eval_dataloader:
            inputs = batch[0].to(self.device)
            label = batch[1].to(self.device)

            with torch.no_grad():
                loss, logit = self.model(inputs, label)

                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

        result = (np.concatenate(logits), np.concatenate(labels))
        result = self.compute_metrics(result)

        print({'eval_' + metric: value for metric, value in result.items()})

        return result[self.metric_for_best_model]

    def predict(self, test_dataset):
        if self.num_devices > 1:
            self.model = nn.DataParallel(self.model)

        self.model.eval()

        test_dataloader = get_dataloader(test_dataset, self.eval_batch_size, self.local_rank)

        logits = []
        for batch in test_dataloader:
            inputs = batch[0].to(self.device)
            label = batch[1].to(self.device)

            with torch.no_grad():
                loss, logit = self.model(inputs, label)
                logits.append(logit.cpu().numpy())

        return np.concatenate(logits)
