# X-MAS: Towards Building Multi-Agent Systems with Heterogeneous LLMs

This repository contains X-MAS-Bench (single-model capability probes) and X-MAS-Design (multi-agent methods). The commands below use the repository's existing `model_dict`/`model_list` configuration format; X-MAS-Design can therefore use multiple model aliases and endpoints in one experiment.

## 1. Install

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m compileall -q X-MAS-Bench X-MAS-Design scripts
```

## 2. Configure models

Edit a private copy of `configs/X-MAS_Bench_config.json` or `configs/X-MAS_Design_config.json`. Each entry keeps the original multi-model schema:

```json
{
  "model_dict": {
    "model-alias": {
      "model_list": [
        {"model_name": "served-model-name", "model_url": "https://server.example/v1", "api_key": "..."}
      ],
      "max_workers_per_model": 1
    }
  }
}
```

For X-MAS-Design, add every model alias referenced by the selected method YAML. For example, `config_reasoner.yaml` references `deepseek-r1-distill-qwen-32b`, while the default configs reference `qwen-2.5-32b-instruct`. The role YAML chooses model aliases; the JSON config supplies their endpoint lists and credentials. Do not commit credentials or private endpoint URLs.

To validate a dataset/config selection without making an API request:

```bash
python X-MAS-Bench/infer_direct.py \
  --model_name qwen-2.5-32b-instruct \
  --model_config configs/X-MAS_Bench_config.json \
  --test_dataset_name AIME-2024 \
  --sample_num 1 \
  --dry_run
```

## 3. Run one direct AIME item

After filling the selected model entry in the private config:

```bash
python X-MAS-Bench/infer_direct.py \
  --model_name qwen-2.5-32b-instruct \
  --model_config configs/X-MAS_Bench_config.json \
  --test_dataset_name AIME-2024 \
  --sample_num 1 \
  --model_max_tokens 8192 \
  --model_temperature 0 \
  --sequential \
  --output_path /tmp/xmas_direct_one.jsonl
```

## 4. Run a MAS method

The method YAML selects role-wise model aliases, while the JSON config supplies the corresponding model endpoints. For example, after adding `deepseek-r1-distill-qwen-32b` to a private Design config:

```bash
python X-MAS-Design/inference_mas.py \
  --method_name llm_debate \
  --method_config_name config_reasoner \
  --model_name deepseek-r1-distill-qwen-32b \
  --model_api_config /path/to/private_X-MAS_Design_config.json \
  --test_dataset_name AIME-2024 \
  --sample_num 1 \
  --model_max_tokens 8192 \
  --model_temperature 0 \
  --sequential \
  --output_path /tmp/xmas_mas_one.jsonl
```

## Full experiments and released results

The original X-MAS-Bench result archive is available from the anonymous Google Drive link supplied with the submission: <https://drive.google.com/file/d/1oukYZLDOuc98i-ICkoZ6OYME9a7-AuH1/view?usp=drive_link>. Place extracted result/dataset folders under the paths expected by the original scripts. The paper's Appendix E specifies the role-to-model configurations for the heterogeneous MAS settings; the YAML/JSON configuration format here provides the corresponding runnable model-alias and endpoint path.

## Dataset provenance and access

The benchmark files are sampled or reformatted third-party evaluation data and remain subject to their upstream licenses and access terms. [DATASETS.md](DATASETS.md) lists the upstream source, paper citation, upstream license/access status, local file, and preparation performed for each of the 23 dataset configurations.

GPQA and GPQA-Diamond are gated by their data provider and are not redistributed in this repository. After obtaining `gpqa_main.csv` and `gpqa_diamond.csv` through the [official GPQA access page](https://huggingface.co/datasets/Idavidrein/gpqa), create the local benchmark files with:

```bash
python scripts/prepare_gpqa.py \
  --main-csv /path/to/gpqa_main.csv \
  --diamond-csv /path/to/gpqa_diamond.csv
```

This command writes the two files under `X-MAS-Bench/benchmarks/`; those generated files are ignored by Git.

## Repository layout

```text
configs/                 model_dict/model_list endpoint aliases
X-MAS-Bench/benchmarks/  benchmark JSON files
X-MAS-Bench/             direct and capability-function inference/evaluation
X-MAS-Design/methods/    LLM-Debate, DyLAN, AgentVerse, and X-MAS-Proto
X-MAS-Design/benchmarks/ benchmark JSON files for MAS runs
scripts/                 evaluation and shell entry points
```

The one-item commands verify installation, configuration, and an executable path; use the Appendix E role mappings and the released result archive when reproducing the paper tables. Keep credentials and private model URLs outside committed files.
