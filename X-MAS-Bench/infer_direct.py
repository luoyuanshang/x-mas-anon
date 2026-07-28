import argparse
import concurrent.futures
import json
import os
import threading
import traceback

from tqdm import tqdm

from config_utils import load_config
from utils import LLM


def parse_args():
    parser = argparse.ArgumentParser(description="Run direct model inference on an X-MAS-Bench dataset.")
    parser.add_argument("--model_name", default="llama-3-70b-instruct")
    parser.add_argument("--model_temperature", type=float, default=0.5)
    parser.add_argument("--model_max_tokens", type=int, default=2048)
    parser.add_argument("--model_timeout", type=int, default=600)
    parser.add_argument("--model_config", default="./configs/X-MAS_Bench_config.json")
    parser.add_argument("--test_dataset_name", default="MedMCQA")
    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and show selected samples without calling a model.")
    return parser.parse_args()


def write_jsonl(lock, path, item):
    with lock:
        with open(path, "a", encoding="utf-8") as result_file:
            result_file.write(json.dumps(item, ensure_ascii=True) + "\n")


def main():
    args = parse_args()
    config = load_config(args.model_config)
    if args.model_name not in config["model_dict"]:
        raise ValueError(f"Model alias {args.model_name!r} is not present in {args.model_config}")

    dataset_path = f"X-MAS-Bench/benchmarks/{args.test_dataset_name}.json"
    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        samples = json.load(dataset_file)
    sample_num = 800 if args.test_dataset_name == "SciKnowEval" else args.sample_num
    samples = samples[:sample_num] if sample_num > 0 else samples

    output_path = args.output_path or f"X-MAS-Bench/results/{args.test_dataset_name}/{args.model_name}_direct.jsonl"
    if args.dry_run:
        print(json.dumps({"status": "validated", "dataset": dataset_path, "samples": len(samples), "output": output_path}, indent=2))
        return 0

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as result_file:
            for line in result_file:
                item = json.loads(line)
                if item.get("status") == "ok":
                    processed.add(item["query"])
    samples = [sample for sample in samples if sample["query"] not in processed]

    model_entry = config["model_dict"][args.model_name]
    model_list = model_entry["model_list"]
    max_workers = model_entry["max_workers_per_model"] * len(model_list)
    lock = threading.Lock()
    failures = []

    def process(sample):
        record = sample.copy()
        try:
            llm = LLM(vars(args), model_list)
            llm.call_llm(prompt=sample["query"])
            metadata = llm.last_call_metadata
            record.update(metadata)
            record["generated_output"] = metadata["final_response"]
            protocol_status = metadata["protocol_status"]
            if protocol_status == "ok":
                record["status"] = "ok"
            elif protocol_status == "token_truncation":
                record["status"] = "token_truncation"
            else:
                record["status"] = "protocol_failure"
            if record["status"] != "ok":
                failures.append(sample["query"])
        except Exception as exc:
            record.update({
                "status": "runtime_failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            failures.append(sample["query"])
        write_jsonl(lock, output_path, record)

    if args.sequential:
        for sample in tqdm(samples, desc="Direct inference"):
            process(sample)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process, samples), total=len(samples), desc="Direct inference"))

    print(json.dumps({"status": "complete", "processed": len(samples), "failures": len(failures), "output": output_path}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
