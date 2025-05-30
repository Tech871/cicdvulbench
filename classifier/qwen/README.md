## Как установить

Получаем токен https://huggingface.co/settings/tokens.
Для `meta-llama/Llama-3.1-8B` нужно выбрать `Read access to contents of all public gated repos you can access`
при создании ключа.

Создаем `.env` в корне проекта и добавляем туда этот токен `HUGGINGFACE_TOKEN`.
Если хотим использовать `S3`, то добавляем токены `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY`.

Для
запуска [Flash Attention на NVIDIA](https://huggingface.co/docs/transformers/perf_infer_gpu_one?install=NVIDIA#flashattention)

```bash
pip install flash-attn --no-build-isolation
```

Правим конфиг `config.json` с описанием того, какие модели, на чем обучать и тестировать

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

Подробнее про версии `Qwen` можно посмотреть здесь https://huggingface.co/Qwen.
Также подойдут `Llama` модели,
такие как `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` или `meta-llama/Llama-3.1-8B`.

Если `num_labels = 2`, то будет применена бинарная классификация - есть уязвимость или ее нет. В остальных случаях
параметр `num_labels` вообще не нужно указывать,
тогда количество классов будет равно количеству таргетов + 1.

Дополнительно можно указать `problem_type`.
Если у каждого примера ровно одна метка, то `single_label_classification`.
Если меток может быть несколько, то `multi_label_classification`.
По умолчанию используется `single_label_classification`.

Чтобы обучение было мультиклассовым,
достаточно в `targets` указать список номеров `CWE`.


## Как запустить на сервере

Убеждаемся, что никто ничего не запускает сейчас

```bash
nvtop
```

Запускаем обучение

```bash
accelerate launch train.py
```

См. [Fully Sharded Data Parallel](https://github.com/huggingface/accelerate/blob/main/docs/source/usage_guides/fsdp.md)

Запускаем тест

```bash
python3 test.py
```

## Как запустить через docker

Собираем или пересобираем образ из корня проекта
```bash
docker build -t имя_образа .
```

Запускаем его с параметрами, переданными через `.env` 
или явно при запуске докера, например, `-e MODEL_PATH=Qwen/Qwen3-0.6B`, 
также можно указать `DEPTH`, `TRAIN` и `TEST`

Для обучения
```bash
docker run --gpus all -it -e RUN=train имя_образа
```
Если хотим продолжить обучение с чекпоинта, то нужно передать его номер, 
например, `-e CHECKPOINT=64`

Для тестирования
```bash
docker run --gpus all -it -e RUN=test имя_образа
```

Для обучения и теста без сохранения на `S3`
```bash
docker run --gpus all -it имя_образа
```

Дополнительно см. `docker-run.sh` в корне проекта.

Удаляем образ, если больше не нужен
```bash
docker rmi -f имя_образа
```

## Как отлаживать

Запускаем `TensorBoard` и пробрасываем порт к себе через `SSH` туннель

```bash
tensorboard --logdir ~/experiment/log/язык/таргет/трейн/классификатор
ssh -L порт:localhost:порт юзернейм@212.41.1.249
```

Либо качаем логи через `scp` и запускаемся у себя

```bash
scp -r юзернейм@212.41.1.249:~/experiment/log/язык/таргет/трейн/классификатор log
tensorboard --logdir log
```

## Как подбирать гиперпараметры

Меняем в `TrainingArguments` параметры, которые хотим подобрать, 
например, `learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2)`.

Меняем количество попыток `n_trials` в `study.optimize` функции,
например, `study.optimize(objective, n_trials=100)`.

```bash
accelerate launch stady.py
```