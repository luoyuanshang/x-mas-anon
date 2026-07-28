# Reproducibility repair notes

This document records changes made after the original anonymous release. It does not replace the submitted supplementary material or revise paper results.

## Confirmed release problems

- The README called `X-MAS-Design/inference_X-MAS.py`, but the repository contains `inference_mas.py`.
- No dependency file or clear clean-environment installation path was provided.
- The planning prompt in `infer_plan.py` was truncated by an unescaped quoted marker.
- Bench plan/revise/evaluate functions read a hard-coded direct-result path that did not match `infer_direct.py` output.
- Sequential MAS inference omitted the required file lock argument.
- The Design evaluation shell script called `eval_mas.py.py` and selected the Bench API config.
- API keys were printed by both LLM clients.
- The legacy benchmark evaluator also printed complete endpoint URLs.
- The requested `--model_max_tokens` value was ignored in favor of a model-name heuristic.
- Network, parser, and other runtime outcomes were not sufficiently visible in the released logs for a reviewer to diagnose the execution path.
- X-MAS-Proto could return an undefined `plans` variable after planner parsing failed.

## Repair behavior

- `requirements.txt` was added, and the config loader now supports optional `${ENV_VAR}` expansion. The normal experiment configs retain the original `model_dict`/`model_list` structure needed for multiple role-wise models and endpoints; environment variables are not a replacement for role-wise model configuration.
- Direct and MAS output rows now record explicit status, raw/final responses, finish reason, token usage, and protocol information.
- When an OpenAI-compatible server exposes DeepSeek reasoning in a separate `reasoning_content` field, that field is retained instead of being discarded; final `content` and `finish_reason` are tracked independently.
- Successful rows alone are treated as completed when resuming a run; failed rows remain retryable.
- Runtime and protocol failures cause a nonzero process exit.
- A model-output parser exception is recorded as `protocol_failure`; network and unexpected code exceptions remain `runtime_failure`.
- The original task-aware evaluator remains in place: structured/code tasks use their existing deterministic or execution-based paths, while the generic path uses the original OpenAI-compatible LLM judge. The repair does not replace the paper evaluator.
- README commands now point to real files, explain the `model_dict`/`model_list` configuration, and include a direct path plus a MAS path.
- Legacy evaluator logs now report endpoint counts instead of private URLs, and output paths without a directory component are supported.

## Validation performed

Run from the repository root:

```bash
python -m compileall -q X-MAS-Bench X-MAS-Design scripts
```

The repaired entry points were also executed against the remote validation environment after installing `requirements.txt` for its Python version. This verifies the installation and inference paths without claiming a new Table 3 result.

## Paper configuration and scope

Appendix E of the paper specifies the role-to-model configurations for the heterogeneous MAS settings. The repaired YAML/JSON path lets a reviewer instantiate those aliases and endpoints without changing the multi-model structure. The repair establishes executable and auditable entry paths; it does not alter the paper's table values. A required intermediate-protocol violation remains an end-to-end MAS failure under the original evaluation definition, even when the response contains a mathematically correct standalone answer.

## Validation environment note

Install `requirements.txt` for the active Python interpreter. A copied binary dependency directory is not portable across Python minor versions: the remote validation host uses Python 3.10, while an earlier temporary `deps/` directory contained Python 3.12 extensions and failed at import time. Reinstalling the requirements for Python 3.10 under `/data/xiangruiliu/xmas_rebuttal_20260728/deps_py310` allowed the repaired entry paths to execute. This temporary remote dependency directory is validation state, not a repository dependency to commit or distribute.
