import torch


class TrainingArguments:
    amp = None

    def __init__(
            self,
            output_dir=None,
            logging_dir=None,
            evaluation_strategy="epoch",
            num_train_epochs=5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            learning_rate=5e-5,
            adam_epsilon=1e-8,
            gradient_accumulation_steps=1,
            max_grad_norm=1,
            metric_for_best_model="f1",
            early_stopping_patience=3,
            early_stopping_threshold=0.0001,
            local_rank=-1,
            opt_level='O1',
            no_cuda=True,
            fp16=False
    ):
        self.output_dir = output_dir
        self.logging_dir = logging_dir

        self.evaluation_strategy = evaluation_strategy

        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.metric_for_best_model = metric_for_best_model
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience = early_stopping_patience

        self.local_rank = local_rank
        self.opt_level = opt_level

        if fp16:
            try:
                from apex import amp
                self.amp = amp
            except ImportError:
                raise ImportError("https://www.github.com/nvidia/apex")

        if no_cuda or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.num_devices = 1
        elif self.local_rank == -1:
            self.device = torch.device("cuda")
            self.num_devices = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group("nccl")
            self.num_devices = 1
