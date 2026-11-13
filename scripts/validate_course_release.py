#!/usr/bin/env python
"""Validate keys, censoring labels, privacy fields, and file hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


FORBIDDEN = {
    "request_id", "sample_id", "pool_prompt_id", "dedup_group",
    "candidate_id", "source_record_id", "normalized_prompt", "prompt",
    "response_text", "output_token_ids", "sampled_token_id",
    "top1_token_id_raw", "prompt_text_sha256", "model_input_ids_sha256",
    "generation_seed", "created_at", "completed_at",
}
LANDMARKS = (32, 64, 128, 256, 512, 1024, 1536)
HORIZONS = (32, 64, 128, 256, 512, 1024)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    data = args.root / "data"
    errors: list[str] = []
    requests = pd.read_csv(data / "requests.csv")
    if len(requests) != 50_000:
        errors.append(f"requests row count {len(requests)} != 50000")
    if requests["case_id"].duplicated().any():
        errors.append("duplicate case_id")
    if not requests["case_id"].str.fullmatch(r"case_[0-9a-f]{16}").all():
        errors.append("invalid case_id format")
    if not requests["prompt_group_id"].str.fullmatch(r"group_[0-9a-f]{16}").all():
        errors.append("invalid prompt_group_id format")
    bad = FORBIDDEN.intersection(requests.columns)
    if bad:
        errors.append(f"forbidden request columns: {sorted(bad)}")
    if not (requests["censored"] == 1 - requests["event"]).all():
        errors.append("censored != 1-event")
    if requests.loc[requests["event"] == 0, "first_eos_token_index"].notna().any():
        errors.append("censored rows have first_eos_token_index")

    request_ids = set(requests["case_id"])
    total_landmarks = 0
    landmark_columns: set[str] | None = None
    for a in LANDMARKS:
        path = data / "landmarks" / f"landmark_{a:04d}.csv"
        frame = pd.read_csv(path)
        total_landmarks += len(frame)
        if set(frame["case_id"]) - request_ids:
            errors.append(f"landmark {a}: unknown case_id")
        if frame["case_id"].duplicated().any():
            errors.append(f"landmark {a}: duplicate case_id")
        if not (frame["landmark"] == a).all():
            errors.append(f"landmark {a}: wrong landmark value")
        if not (frame["remaining_followup_tokens"] > 0).all():
            errors.append(f"landmark {a}: nonpositive remaining follow-up")
        if FORBIDDEN.intersection(frame.columns):
            errors.append(f"landmark {a}: forbidden columns")
        for h in HORIZONS:
            target = frame[f"end_within_{h}"]
            observed = frame[f"end_within_{h}_observed"]
            if target[observed == 0].notna().any():
                errors.append(f"landmark {a}: unobserved horizon {h} has labels")
            if target[observed == 1].isna().any():
                errors.append(f"landmark {a}: observed horizon {h} lacks labels")
            if not target.dropna().isin([0, 1]).all():
                errors.append(f"landmark {a}: invalid horizon {h} label")
        cols = set(frame.columns)
        if landmark_columns is None:
            landmark_columns = cols
        elif cols != landmark_columns:
            errors.append(f"landmark {a}: schema differs")

    if total_landmarks != 245_833:
        errors.append(f"landmark row count {total_landmarks} != 245833")

    dictionary = pd.read_csv(data / "data_dictionary.csv")
    covered_requests = set(dictionary.loc[dictionary["table"] == "requests.csv", "column"])
    covered_landmarks = set(dictionary.loc[dictionary["table"] == "landmarks/*.csv", "column"])
    if covered_requests != set(requests.columns):
        errors.append("data dictionary does not exactly cover requests columns")
    if landmark_columns is not None and covered_landmarks != landmark_columns:
        errors.append("data dictionary does not exactly cover landmark columns")

    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = data / item["path"]
        if not path.exists() or sha256(path) != item["sha256"]:
            errors.append(f"hash mismatch: {item['path']}")

    result = {
        "valid": not errors,
        "errors": errors,
        "request_rows": len(requests),
        "landmark_rows": total_landmarks,
        "prompt_groups": int(requests["prompt_group_id"].nunique()),
        "events": int(requests["event"].sum()),
        "censored": int(requests["censored"].sum()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
