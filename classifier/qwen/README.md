
## How to Install

Get a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
For `meta-llama/Llama-3.1-8B`, you need to select `Read access to contents of all public gated repos you can access` when creating the key.

Create a `.env` file in the root of the project and add this token as `HUGGINGFACE_TOKEN`.
If you want to use `S3`, also add the tokens `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.

To run [Flash Attention on NVIDIA](https://huggingface.co/docs/transformers/perf_infer_gpu_one?install=NVIDIA#flashattention):

```bash
pip install flash-attn --no-build-isolation
```

Edit the `config.json` file to describe which models to train and test, and on what data.

```json
{
  "classifiers": [
    {
      "pretrained_model_path": "Qwen/Qwen2.5-Coder-1.5B",
      "num_labels": 2,
      "problem_type": "single_label_classification"
    }
  ],
  "trains": [
    {
      "train": "crawler",
      "eval": "osv_dev"
    },
    "maven"
  ],
  "tests": [
    "cve_fixes"
  ],
  "languages": [
    "java"
  ],
  "targets": [
    502,
    [
      20,
      79
    ]
  ]
}
```

More information about `Qwen` versions can be found here: [https://huggingface.co/Qwen](https://huggingface.co/Qwen).
Llama models are also suitable, such as `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` or `meta-llama/Llama-3.1-8B`.

If `num_labels = 2`, binary classification will be applied—whether there is a vulnerability or not. In other cases, you do not need to specify the `num_labels` parameter; then, the number of classes will equal the number of targets + 1.

Additionally, you can specify `problem_type`.
If each sample has exactly one label, use `single_label_classification`.
If there can be several labels, use `multi_label_classification`.
By default, `single_label_classification` is used.

To make training multiclass, just specify a list of `CWE` numbers in `targets`.

## How to Run on the Server

Make sure no one else is running anything at the moment:

```bash
nvtop
```

Start training:

```bash
accelerate launch train.py
```

See [Fully Sharded Data Parallel](https://github.com/huggingface/accelerate/blob/main/docs/source/usage_guides/fsdp.md)

Start testing:

```bash
python3 test.py
```

## How to Run with Docker

Build or rebuild the image from the root of the project:

```bash
docker build -t image_name .
```

Run it with parameters passed via `.env` or explicitly when starting the container, for example, `-e MODEL_PATH=Qwen/Qwen3-0.6B`.
You can also specify `DEPTH`, `TRAIN`, and `TEST`.

For training:

```bash
docker run --gpus all -it -e RUN=train image_name
```

If you want to resume training from a checkpoint, pass its number, e.g., `-e CHECKPOINT=64`

For testing:

```bash
docker run --gpus all -it -e RUN=test image_name
```

For training and testing without saving to `S3`:

```bash
docker run --gpus all -it image_name
```

Additionally, see `docker-run.sh` in the root of the project.

Delete the image if you no longer need it:

```bash
docker rmi -f image_name
```

## How to Debug

Start `TensorBoard` and forward the port to yourself via an `SSH` tunnel:

```bash
tensorboard --logdir ~/experiment/log/language/target/train/classifier
ssh -L port:localhost:port username@212.41.1.249
```

Or download the logs via `scp` and run locally:

```bash
scp -r username@212.41.1.249:~/experiment/log/language/target/train/classifier log
tensorboard --logdir log
```

## How to Select Hyperparameters

Change in `TrainingArguments` the parameters you want to tune,
for example, `learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2)`.

Change the number of attempts `n_trials` in the `study.optimize` function,
for example, `study.optimize(objective, n_trials=100)`.

```bash
accelerate launch study.py
```


