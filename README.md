# CICD-Vul: Benchmarking Large Language Models with Context Variations

A comprehensive benchmark for evaluating Large Language Models (LLMs) on vulnerability detection in CI/CD pipeline contexts, featuring realistic code samples with varying contextual complexity.

## 🔍 Overview

CICD-Vul addresses a critical gap in vulnerability detection research by providing a benchmark that evaluates LLMs under realistic CI/CD conditions with varying contextual information. Unlike existing benchmarks that focus on isolated code snippets, our benchmark includes surrounding code context that mirrors real-world development scenarios.

### Key Features

- **Real-world vulnerabilities** from 528 open-source repositories
- **Context variations** across 6 depth levels (0-5)
- **15 CWE vulnerability types** from the CWE Top 25 list
- **Expert-validated annotations** with 84% precision
- **Baseline comparisons** including tf-idf and multiple LLM variants

### Main Findings

- Context depth significantly impacts model performance
- Current LLMs struggle with practical vulnerability detection (best MCC: 0.057)

## 📊 Dataset

### Training Dataset
- **19,629 commits** from real-world repositories
- **17 CWE classes** with expert validation
- **Multi-stage filtering** from 4.9M initial commits
- **84% annotation precision** after expert review

### Test Dataset
- **1,179 Java commits** spanning 15 CWE types
- **1,808 revisions** from 528 repositories
- **CVE-based ground truth** from OSV dataset


## 🧪 Experimental Setup

### Models Evaluated

1. **tf-idf** - Traditional bag-of-words baseline
2. **Qwen2.5-Coder-1.5b** - Small code-specific LLM
3. **Qwen2.5-Coder-7b** - Medium code-specific LLM  
4. **Qwen2.5-Coder-32b** - Large code-specific LLM
5. **Qwen3-8b** - General-purpose LLM

### Context Depths

- **Depth 0**: Function only
- **Depth 1**: Function + immediate callers
- **Depth 2**: Function + 2-level call context
- **Depth 3**: Function + 3-level call context
- **Depth 5**: Function + 5-level call context

### Evaluation Metrics

- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **MCC**: Matthews Correlation Coefficient (primary metric)



## Launch instructions

### Qwen models

1. Unpack data.tar.gz either to local file system or to S3 bucket.
In the latter case, ensure that `src` is a top-level directory.

2. Create .env file with following content:

```
AWS_ACCESS_KEY_ID=<key id>
AWS_SECRET_ACCESS_KEY=<key>
HUGGINGFACE_TOKEN=<token>
CLASSIFIER=<folder name to save checkpoints and data>
MODEL_PATH=<path of the model in Hugginface>
BUCKET=<S3 bucket to work with; do not set if you want to run locally>
export S3_ENDPOINT=<S3 service endpoint if we want to keep data and checkpoints on S3, don't set otherwise>
```

if you want to run evaluation with prepared checkpoints, add
```
RUN=test
```

If you want to obtain checkpoints, but run no tests, add

If you want to evaluate pre-trained models, add
```
RUN=test
export PRETRAINED=1
```

If you run scripts w/o S3, add
```
export ROOT_DIR=<directory where src from data.tar.gz is>
```

3. Build a Docker image and execute
```
for d in 0 1 2 3 5; do docker run --gpus all -it -e DEPTH=$d 7bbd48ab94cc; done
```

### Tf-idf classifier and misc models

Unpack data.tar.gz. Either export `ROOT_DIR` as specified above or
move `src` to working directory.

Follow instructions from classifier/<name>/README.md then.


