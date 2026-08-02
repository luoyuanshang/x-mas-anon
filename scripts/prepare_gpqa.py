#!/usr/bin/env python3
"""Prepare local X-MAS GPQA files from user-obtained official CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


SEED = 2024
REQUIRED_COLUMNS = {
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
    "High-level domain",
    "Subdomain",
    "Writer's Difficulty Estimate",
}


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Convert official, user-obtained GPQA CSV files to the local X-MAS "
            "benchmark JSON format."
        )
    )
    parser.add_argument("--main-csv", type=Path, required=True)
    parser.add_argument("--diamond-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "X-MAS-Bench" / "benchmarks",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing local GPQA JSON files.",
    )
    return parser.parse_args()


def read_official_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Official GPQA CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        available = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - available)
        if missing:
            raise ValueError(
                f"{path} is missing official GPQA columns: {', '.join(missing)}"
            )
        return list(reader)


def format_records(rows: list[dict[str, str]], dataset_name: str) -> list[dict]:
    records = []
    for row in rows:
        query = (
            f"{row['Question']}\n\n"
            "Choose the correct answer from the following options:"
            f"\n(A) {row['Correct Answer']}"
            f"\n(B) {row['Incorrect Answer 1']}"
            f"\n(C) {row['Incorrect Answer 2']}"
            f"\n(D) {row['Incorrect Answer 3']}"
        )
        records.append(
            {
                "query": query,
                "gt": f"(A) {row['Correct Answer']}",
                "tag": [
                    dataset_name,
                    row["High-level domain"],
                    row["Subdomain"],
                    row["Writer's Difficulty Estimate"],
                ],
                "source": dataset_name,
            }
        )

    random.Random(SEED).shuffle(records)
    return records


def write_json(records: list[dict], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace {output_path}; pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    print(f"Wrote {len(records)} records to {output_path}")


def main() -> None:
    args = parse_args()
    jobs = (
        (args.main_csv, "GPQA", args.output_dir / "GPQA.json"),
        (
            args.diamond_csv,
            "GPQA-Diamond",
            args.output_dir / "GPQA-Diamond.json",
        ),
    )
    for csv_path, dataset_name, output_path in jobs:
        records = format_records(read_official_csv(csv_path), dataset_name)
        write_json(records, output_path, args.overwrite)


if __name__ == "__main__":
    main()
