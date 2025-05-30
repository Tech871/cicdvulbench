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
- **Balanced distribution** across vulnerability types


## 🚀 Installation



### Setup

```bash
ADD SCRIPT

```

## ⚡ Quick Start

### Evaluation

```python
ADD SCRIPT

```

### Run Full Benchmark

```bash
# Evaluate all models across all depths
python scripts/run_benchmark.py --config configs/full_evaluation.yaml

# Generate results table
python scripts/generate_results.py --output results/table6.csv
```

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

## 📈 Results

### Best Model Performance (Table 6)

| Model | Depth | Precision | Recall | MCC |
|-------|-------|-----------|---------|-----|
| tf-idf | 1 | **1.000** | 0.073 | **0.057** |
| Qwen2.5-Coder-32b | 2 | 1.000 | 0.050 | 0.047 |
| Qwen2.5-Coder-32b | 3 | 1.000 | 0.063 | 0.030 |


## 🔄 Reproducibility

All experiments are fully reproducible:

```bash

```

## Launch instructions

Create .env file with following content:

```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
HUGGINGFACE_TOKEN=
BUCKET=
CLASSIFIER=
MODEL_PATH=
DEPTH=
```

where `BUCKET` is a name of AWS bucket to use for input / output data,
`MODEL_PATH` is a path of a model on Huggingface,
make a Docker image and run the container.

