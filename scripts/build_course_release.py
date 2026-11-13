#!/usr/bin/env python
"""Build the anonymized, analysis-ready CSV course release."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
from pathlib import Path

import numpy as np
import pandas as pd


LANDMARKS = (32, 64, 128, 256, 512, 1024, 1536)
HORIZONS = (32, 64, 128, 256, 512, 1024)
ENDPOINT_UPPERS = np.array(
    [32, 64, 128, 256, 384, 512, 640, 768, 896, 1024,
     1152, 1280, 1408, 1536, 1664, 1792, 1920]
)

STATIC_COLUMNS = [
    "prompt_tokens", "log1p_prompt_tokens", "was_truncated",
    "language_detected", "language_confidence", "language_second",
    "language_margin", "language_status", "code_switching",
    "code_switching_score", "length_instruction_type",
    "has_numeric_length_request", "requested_length_value",
    "requested_length_lower", "requested_length_upper",
    "requested_length_unit", "log1p_requested_length_value",
    "has_code_fence", "code_fence_count", "has_url", "url_count",
    "has_bullet_list", "bullet_line_count", "has_valid_json",
    "has_json_like_structure", "paragraph_count", "question_mark_count",
    "digit_count", "digit_ratio", "repetition_bigram_ratio",
]

OUTCOME_COLUMNS = [
    "model_input_tokens", "batch_size_actual", "output_tokens_excluding_eos",
    "survival_time_including_eos", "event", "finish_reason",
    "first_eos_token_index", "analysis_eligible", "time_to_first_token_ms",
    "total_latency_ms", "max_new_tokens",
]

DYNAMIC_COLUMNS = [
    "eos_logprob_last", "eos_logprob_mean", "eos_logprob_max",
    "eos_logprob_slope", "log_eos_rank_last", "log_eos_rank_mean",
    "log_eos_rank_min", "log_eos_rank_slope", "entropy_raw_last",
    "entropy_raw_mean", "entropy_raw_slope", "margin_last", "margin_mean",
    "margin_slope", "surprisal_mean", "surprisal_slope", "support_rate",
    "processed_entropy_mean", "repetition_last", "repetition_mean",
    "repetition_slope", "sentence_rate", "paragraph_rate",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_or_create_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_hex(32), encoding="ascii")
    return bytes.fromhex(path.read_text(encoding="ascii").strip())


def pseudonym(key: bytes, namespace: str, value: str, prefix: str) -> str:
    payload = f"{namespace}:{value}".encode("utf-8")
    return prefix + hmac.new(key, payload, hashlib.sha256).hexdigest()[:16]


def endpoint_interval(t: pd.Series) -> tuple[np.ndarray, list[str]]:
    classes = np.searchsorted(ENDPOINT_UPPERS, t.to_numpy(), side="left")
    lowers = np.r_[1, ENDPOINT_UPPERS[:-1] + 1]
    labels = [f"{lo}-{hi}" for lo, hi in zip(lowers, ENDPOINT_UPPERS)] + [">1920"]
    return classes, [labels[i] for i in classes]


def language_group(series: pd.Series) -> pd.Series:
    major = {"en", "zh", "ru", "fr", "pt", "es", "de", "it", "tr", "ja"}
    return series.where(series.isin(major), "other")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.10g", lineterminator="\n")


def build_dictionary(request_columns: list[str], landmark_columns: list[str]) -> pd.DataFrame:
    descriptions = {
        "case_id": "HMAC-pseudonymous generation request key",
        "prompt_group_id": "HMAC-pseudonymous repeated-prompt cluster key",
        "sampling_block": "One of ten independent with-replacement sampling blocks",
        "split": "Frozen group-safe train/validation/test split",
        "prompt_tokens": "Prompt tokens before chat template",
        "model_input_tokens": "Model input tokens after chat template",
        "log1p_prompt_tokens": "Natural log of one plus prompt_tokens",
        "was_truncated": "Prompt truncated at 4096 tokenizer tokens",
        "language_detected": "Heuristic primary language label",
        "language_group": "Major language or other",
        "language_confidence": "Heuristic language confidence",
        "language_second": "Second-ranked heuristic language",
        "language_margin": "Gap between top two language scores",
        "language_status": "Language detector status",
        "code_switching": "Heuristic multi-language prompt flag",
        "code_switching_score": "Heuristic code-switching score",
        "length_instruction_type": "Rule-based requested-length instruction type",
        "has_numeric_length_request": "Prompt contains a numeric length request",
        "requested_length_value": "Parsed central requested length",
        "requested_length_lower": "Parsed requested-length lower bound",
        "requested_length_upper": "Parsed requested-length upper bound",
        "requested_length_unit": "Parsed requested-length unit",
        "log1p_requested_length_value": "Natural log of one plus requested length",
        "survival_time": "Token risk time including final EOS for events",
        "event": "1 if EOS observed; 0 if administratively censored",
        "censored": "1-event",
        "censor_limit": "Administrative maximum new-token limit",
        "output_tokens_excluding_eos": "Visible output tokens",
        "finish_reason": "eos or length",
        "first_eos_token_index": "EOS token index; missing when censored",
        "analysis_eligible": "Passes the frozen main-analysis eligibility rules",
        "endpoint_class_18": "Zero-based fixed terminal-length class",
        "endpoint_interval_18": "Human-readable fixed terminal-length interval",
        "time_to_first_token_ms": "Batch-context time to first token in milliseconds",
        "total_latency_ms": "Batch-lifecycle request latency in milliseconds",
        "batch_size_actual": "Dynamic batch size when request entered generation",
        "landmark": "Tokens generated before the landmark forecast",
        "remaining_followup_tokens": "Observed follow-up tokens after landmark",
        "remaining_event": "EOS event indicator for remaining follow-up",
        "remaining_censor_limit": "Maximum observable follow-up after landmark",
    }
    for c in DYNAMIC_COLUMNS:
        descriptions[c] = "Online dynamic feature; see FIELD_DICTIONARY.md"
    for h in HORIZONS:
        descriptions[f"end_within_{h}"] = f"EOS within {h} tokens after landmark; blank if unknown"
        descriptions[f"end_within_{h}_observed"] = f"Whether end_within_{h} is identifiable"
    structural = {
        "has_code_fence", "has_url", "has_bullet_list", "has_valid_json",
        "has_json_like_structure", "code_fence_count", "url_count",
        "bullet_line_count", "paragraph_count", "question_mark_count",
        "digit_count", "digit_ratio", "repetition_bigram_ratio",
    }
    for c in structural:
        descriptions.setdefault(c, "Rule-based prompt structure feature")
    rows = []
    for table, columns in [("requests.csv", request_columns), ("landmarks/*.csv", landmark_columns)]:
        for c in columns:
            missing = "allowed when not applicable or not observable" if c.startswith("requested_") or c.startswith("end_within_") or c == "first_eos_token_index" else "not expected"
            rows.append({"table": table, "column": c, "description": descriptions.get(c, "See project methods and source code"), "missing_values": missing})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "landmarks").mkdir(exist_ok=True)
    key = load_or_create_key(args.key_file)

    labels = pd.read_parquet(args.labels)
    inputs = pd.read_parquet(args.inputs)[["request_id", "split"]]
    request_files = sorted(args.request_dir.glob("part-*.parquet"))
    if not request_files:
        raise FileNotFoundError(f"No request parquet files in {args.request_dir}")
    outcomes = pd.concat(
        [pd.read_parquet(p, columns=["request_id", *OUTCOME_COLUMNS]) for p in request_files],
        ignore_index=True,
    )
    if any(x["request_id"].duplicated().any() for x in (labels, inputs, outcomes)):
        raise ValueError("Duplicate request_id in source tables")
    merged = labels.merge(inputs, on="request_id", validate="one_to_one")
    merged = merged.merge(outcomes, on="request_id", validate="one_to_one")
    if len(merged) != 50_000:
        raise ValueError(f"Expected 50000 requests, found {len(merged)}")

    merged["case_id"] = [pseudonym(key, "request", x, "case_") for x in merged["request_id"]]
    merged["prompt_group_id"] = [pseudonym(key, "prompt", x, "group_") for x in merged["dedup_group"]]
    merged["language_group"] = language_group(merged["language_detected"])
    merged["survival_time"] = merged["survival_time_including_eos"].astype(int)
    merged["censored"] = (~merged["event"].astype(bool)).astype(int)
    merged["censor_limit"] = merged["max_new_tokens"].astype(int)
    classes, intervals = endpoint_interval(merged["survival_time"])
    merged["endpoint_class_18"] = classes
    merged["endpoint_interval_18"] = intervals
    merged["sampling_block"] = merged["subset_id"].str.replace("subset_", "block_", regex=False)

    request_columns = [
        "case_id", "prompt_group_id", "sampling_block", "split",
        "prompt_tokens", "model_input_tokens", "log1p_prompt_tokens",
        "was_truncated", "language_detected", "language_group",
        "language_confidence", "language_second", "language_margin",
        "language_status", "code_switching", "code_switching_score",
        "length_instruction_type", "has_numeric_length_request",
        "requested_length_value", "requested_length_lower",
        "requested_length_upper", "requested_length_unit",
        "log1p_requested_length_value", "has_code_fence", "code_fence_count",
        "has_url", "url_count", "has_bullet_list", "bullet_line_count",
        "has_valid_json", "has_json_like_structure", "paragraph_count",
        "question_mark_count", "digit_count", "digit_ratio",
        "repetition_bigram_ratio", "output_tokens_excluding_eos",
        "survival_time", "event", "censored", "censor_limit",
        "finish_reason", "first_eos_token_index", "endpoint_class_18",
        "endpoint_interval_18", "analysis_eligible", "batch_size_actual",
        "time_to_first_token_ms", "total_latency_ms",
    ]
    requests_public = merged[request_columns].sort_values("case_id").reset_index(drop=True)
    for c in ["event", "was_truncated", "code_switching", "has_numeric_length_request",
              "has_code_fence", "has_url", "has_bullet_list", "has_valid_json",
              "has_json_like_structure", "analysis_eligible"]:
        requests_public[c] = requests_public[c].astype(int)
    write_csv(requests_public, args.output / "requests.csv")

    features = pd.read_parquet(args.landmarks)
    if features.duplicated(["request_id", "landmark"]).any():
        raise ValueError("Duplicate request_id/landmark rows")
    base = merged[["request_id", "case_id", "prompt_group_id", "split", "survival_time", "event", "censored", "censor_limit", "endpoint_class_18", "endpoint_interval_18"]]
    panel = features.merge(base, on="request_id", validate="many_to_one")
    if not (panel["survival_time"] > panel["landmark"]).all():
        raise ValueError("Landmark panel contains requests not at risk")
    panel["remaining_followup_tokens"] = panel["survival_time"] - panel["landmark"]
    panel["remaining_event"] = panel["event"].astype(int)
    panel["remaining_censor_limit"] = panel["censor_limit"] - panel["landmark"]
    for h in HORIZONS:
        observable = panel["event"].astype(bool) | (panel["remaining_censor_limit"] >= h)
        target = (panel["event"].astype(bool) & (panel["remaining_followup_tokens"] <= h)).astype("Int8")
        target = target.mask(~observable)
        panel[f"end_within_{h}"] = target
        panel[f"end_within_{h}_observed"] = observable.astype(int)

    landmark_columns = [
        "case_id", "prompt_group_id", "split", "landmark",
        "remaining_followup_tokens", "remaining_event", "remaining_censor_limit",
        "endpoint_class_18", "endpoint_interval_18", *DYNAMIC_COLUMNS,
    ]
    for h in HORIZONS:
        landmark_columns.extend([f"end_within_{h}", f"end_within_{h}_observed"])

    summaries = []
    summaries.extend(
        {"table": "requests", "slice": split, "rows": len(g), "events": int(g["event"].sum()), "censored": int(g["censored"].sum())}
        for split, g in requests_public.groupby("split", sort=True)
    )
    for landmark in LANDMARKS:
        frame = panel[panel["landmark"] == landmark][landmark_columns].sort_values("case_id")
        path = args.output / "landmarks" / f"landmark_{landmark:04d}.csv"
        write_csv(frame, path)
        for split, group in frame.groupby("split", sort=True):
            summaries.append({"table": f"landmark_{landmark:04d}", "slice": split, "rows": len(group), "events": int(group["remaining_event"].sum()), "censored": int((1 - group["remaining_event"]).sum())})

    dictionary = build_dictionary(request_columns, landmark_columns)
    write_csv(dictionary, args.output / "data_dictionary.csv")
    write_csv(pd.DataFrame(summaries), args.output / "dataset_summary.csv")

    files = sorted(p for p in args.output.rglob("*.csv"))
    manifest = {
        "release": "course-csv-v1.0.0",
        "privacy_profile": "no text, token ids, source ids, raw hashes, or seeds",
        "rows": {"requests": len(requests_public), "landmarks": len(panel)},
        "landmark_counts": {str(int(k)): int(v) for k, v in panel["landmark"].value_counts().sort_index().items()},
        "files": [{"path": str(p.relative_to(args.output)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in files],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["rows"] | {"csv_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
