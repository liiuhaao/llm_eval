# LLM EVAL

## Tasks

- **Math Reasoning**: GSM8K, MGSM, AIME, AMC, MathQA
- **Science QA**: SciQ, GPQA, OpenBookQA, ARC
- **Language Understanding**: BBH, MMLU-Pro, Winogrande, SocialIQA, TruthfulQA
- **Medical QA**: MedMCQA

## Quick Start

### Install Dependencies
```bash
# Using pip
pip install -e .

# Or using uv
uv pip install -e .
```

### Run Evaluation
```bash
# Run on all models and all tasks
python main.py --batch_size 16

# Run on specific task
python main.py --task GSM8K --batch_size 16

# Run on specific model
python main.py --model Qwen/Qwen3-8B --batch_size 16
```

### Main Arguments
- `--task`: Specify the evaluation task (e.g., GSM8K, MMLU-Pro; if not specified, runs on all datasets)
- `--model`: Specify the model to evaluate (if not specified, runs on all models in models.csv)
- `--batch_size`: Batch size for inference (default: 16)
- `--output_file`: Output file for results (default: data/data.jsonl)
- `--gpu_memory_utilization`: GPU memory utilization ratio (default: 0.6)
- `--local_dir`: Local directory for downloading models (default: /home/liuhao/huggingface)
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 1)
- `--temperature`: Sampling temperature (default: 0.0)
- `--n`: Number of completions to generate per prompt (default: 1)
- `--overwrite`: Overwrite existing results instead of resuming from checkpoint