#!/usr/bin/env python3
"""Evaluate and hill-climb image gating prompts against labeled data."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from openai import OpenAI

from AnkiDeckToImages import parse_gating_decision
from config import GATING_PROMPT_ID, GATING_PROMPT_VERSION


@dataclass
class Example:
    front: str
    back: str
    label: bool
    source: dict[str, Any]


def parse_bool_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        raise ValueError("Missing label value.")
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unrecognized boolean label: {value!r}")


def _first_value(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def parse_example_row(row: dict[str, Any]) -> Example:
    front = _first_value(row, ["front", "Front", "foreign", "korean"])
    back = _first_value(row, ["back", "Back", "english", "meaning"])
    label_raw = None
    for key in ("label", "should_generate", "generate_image", "target"):
        if key in row:
            label_raw = row[key]
            break
    if not front or not back:
        raise ValueError("Row must include front/back text.")
    label = parse_bool_label(label_raw)
    return Example(front=front, back=back, label=label, source=row)


def load_examples(path: Path) -> list[Example]:
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []
    if suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    elif suffix == ".json":
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict) and isinstance(parsed.get("examples"), list):
            rows = [x for x in parsed["examples"] if isinstance(x, dict)]
        elif isinstance(parsed, list):
            rows = [x for x in parsed if isinstance(x, dict)]
        else:
            raise ValueError("JSON must be a list of rows or {'examples': [...]} object.")
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
    else:
        raise ValueError("Supported dataset formats: .jsonl, .json, .csv")
    return [parse_example_row(row) for row in rows]


def confusion_counts(y_true: list[bool], y_pred: list[bool]) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif not truth and not pred:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, float | int]:
    counts = confusion_counts(y_true, y_pred)
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, max(1, len(y_true)))
    return {
        **counts,
        "total": len(y_true),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def response_text(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if output_text:
        return output_text
    output = getattr(resp, "output", None) or []
    parts: list[str] = []
    for item in output:
        for content in getattr(item, "content", []):
            maybe_text = getattr(content, "text", None)
            if maybe_text:
                parts.append(maybe_text)
    return "".join(parts)


def judge_prompt_ref(client: OpenAI, ex: Example, prompt_id: str, prompt_version: str) -> bool:
    resp = client.responses.create(
        prompt={
            "id": prompt_id,
            "version": prompt_version,
            "variables": {"front": ex.front, "back": ex.back},
        }
    )
    return parse_gating_decision(response_text(resp))


def judge_inline(client: OpenAI, ex: Example, model: str, candidate_prompt: str) -> bool:
    rendered = candidate_prompt.format(front=ex.front, back=ex.back)
    resp = client.responses.create(model=model, input=rendered)
    return parse_gating_decision(response_text(resp))


def evaluate_examples(
    examples: list[Example],
    *,
    client: OpenAI,
    mode: str,
    prompt_id: str,
    prompt_version: str,
    model: str,
    candidate_prompt: str,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    preds: list[bool] = []
    labels: list[bool] = []
    rows: list[dict[str, Any]] = []
    for ex in examples:
        if mode == "prompt-ref":
            pred = judge_prompt_ref(client, ex, prompt_id, prompt_version)
        else:
            pred = judge_inline(client, ex, model, candidate_prompt)
        preds.append(pred)
        labels.append(ex.label)
        rows.append(
            {
                "front": ex.front,
                "back": ex.back,
                "label": ex.label,
                "prediction": pred,
                "correct": pred == ex.label,
            }
        )
    return rows, metrics(labels, preds)


def parse_variants(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("variants", [])
        if isinstance(payload, list):
            return [str(x).strip() for x in payload if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return [line.strip("- ").strip() for line in text.splitlines() if line.strip()]


def propose_variants(
    client: OpenAI,
    *,
    model: str,
    seed_prompt: str,
    branching: int,
) -> list[str]:
    request = (
        "Create improved variants of this binary gating prompt for deciding whether "
        "an Anki vocab pair should get an image. Output strict JSON only: "
        '{"variants":["..."]}. Keep placeholders {front} and {back}. '
        f"Return exactly {branching} variants.\n\nSeed prompt:\n{seed_prompt}"
    )
    resp = client.responses.create(model=model, input=request)
    variants = parse_variants(response_text(resp))
    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant not in seen:
            deduped.append(variant)
            seen.add(variant)
    return deduped[:branching]


def score_value(summary: dict[str, float | int], objective: str) -> float:
    return float(summary.get(objective, 0.0))


def run_eval(args: argparse.Namespace) -> int:
    examples = load_examples(Path(args.dataset))
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    candidate_prompt = ""
    if args.mode == "inline":
        candidate_prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    rows, summary = evaluate_examples(
        examples,
        client=client,
        mode=args.mode,
        prompt_id=args.prompt_id,
        prompt_version=args.prompt_version,
        model=args.model,
        candidate_prompt=candidate_prompt,
    )

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["front", "back", "output"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "front": row["front"],
                        "back": row["back"],
                        "output": str(bool(row["prediction"])).lower(),
                    }
                )

    print(json.dumps({"ok": True, "mode": args.mode, "summary": summary}, ensure_ascii=False))
    return 0


def run_hillclimb(args: argparse.Namespace) -> int:
    examples = load_examples(Path(args.dataset))
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    seed_prompt = Path(args.seed_prompt).read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    beam: list[tuple[str, dict[str, float | int]]] = []
    _, seed_summary = evaluate_examples(
        examples,
        client=client,
        mode="inline",
        prompt_id="",
        prompt_version="",
        model=args.judge_model,
        candidate_prompt=seed_prompt,
    )
    beam.append((seed_prompt, seed_summary))

    leaderboard: list[dict[str, Any]] = [
        {"round": 0, "prompt": seed_prompt, "summary": seed_summary}
    ]

    for round_idx in range(1, args.rounds + 1):
        candidates: list[str] = []
        for prompt_text, _ in beam:
            variants = propose_variants(
                client,
                model=args.mutation_model,
                seed_prompt=prompt_text,
                branching=args.branching,
            )
            candidates.extend(variants)
        if not candidates:
            break
        candidates = list(dict.fromkeys(candidates))
        scored: list[tuple[str, dict[str, float | int]]] = []
        for candidate in candidates:
            _, summary = evaluate_examples(
                examples,
                client=client,
                mode="inline",
                prompt_id="",
                prompt_version="",
                model=args.judge_model,
                candidate_prompt=candidate,
            )
            scored.append((candidate, summary))
            leaderboard.append({"round": round_idx, "prompt": candidate, "summary": summary})
        scored.sort(
            key=lambda item: (
                score_value(item[1], args.objective),
                score_value(item[1], "accuracy"),
            ),
            reverse=True,
        )
        beam = scored[: args.keep_top]

    best_prompt, best_summary = beam[0]
    result = {
        "ok": True,
        "objective": args.objective,
        "best_summary": best_summary,
        "best_prompt_file": str(out_dir / "best_prompt.txt"),
        "leaderboard_file": str(out_dir / "leaderboard.jsonl"),
    }
    (out_dir / "best_prompt.txt").write_text(best_prompt, encoding="utf-8")
    with (out_dir / "leaderboard.jsonl").open("w", encoding="utf-8") as handle:
        for row in leaderboard:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


def run_upload_dataset(args: argparse.Namespace) -> int:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    dataset = Path(args.dataset)
    with dataset.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="assistants")
    print(
        json.dumps(
            {
                "ok": True,
                "file_id": uploaded.id,
                "filename": uploaded.filename,
                "bytes": uploaded.bytes,
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_make_template(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"front": "성격", "back": "personality", "label": False},
        {"front": "웃다", "back": "to smile", "label": True},
        {"front": "버스 정류장", "back": "bus stop", "label": True},
        {"front": "그리고", "back": "and", "label": False},
        {"front": "행복하다", "back": "to be happy", "label": True},
    ]
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"ok": True, "wrote": str(out), "rows": len(rows)}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image gating prompt eval + hill-climb.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_template = sub.add_parser("make-template", help="Create a starter labeled dataset.")
    p_template.add_argument(
        "--out",
        default="datasets/image_gating_ground_truth_template.jsonl",
        help="Output path for dataset template (.jsonl).",
    )
    p_template.set_defaults(func=run_make_template)

    p_upload = sub.add_parser("upload-dataset", help="Upload labeled dataset to OpenAI Files.")
    p_upload.add_argument("--dataset", required=True, help="Path to labeled dataset file.")
    p_upload.set_defaults(func=run_upload_dataset)

    p_eval = sub.add_parser("eval", help="Run gating evaluation on labeled data.")
    p_eval.add_argument("--dataset", required=True, help="Dataset path (.jsonl/.json/.csv).")
    p_eval.add_argument(
        "--mode",
        choices=["prompt-ref", "inline"],
        default="prompt-ref",
        help="Use stored prompt id/version or inline candidate prompt text.",
    )
    p_eval.add_argument("--prompt-id", default=GATING_PROMPT_ID)
    p_eval.add_argument("--prompt-version", default=GATING_PROMPT_VERSION)
    p_eval.add_argument("--model", default="gpt-4.1-mini")
    p_eval.add_argument("--prompt-file", help="Inline prompt file (required for --mode inline).")
    p_eval.add_argument(
        "--out-csv",
        default="tmp/image_gating_eval_outputs.csv",
        help="CSV output path with front,back,output judge decisions.",
    )
    p_eval.set_defaults(func=run_eval)

    p_hill = sub.add_parser("hillclimb", help="Hill-climb inline candidate prompts.")
    p_hill.add_argument("--dataset", required=True, help="Dataset path (.jsonl/.json/.csv).")
    p_hill.add_argument("--seed-prompt", required=True, help="Seed prompt file for candidate.")
    p_hill.add_argument("--judge-model", default="gpt-4.1-mini")
    p_hill.add_argument("--mutation-model", default="gpt-4.1-mini")
    p_hill.add_argument("--rounds", type=int, default=2)
    p_hill.add_argument("--branching", type=int, default=4)
    p_hill.add_argument("--keep-top", type=int, default=2)
    p_hill.add_argument(
        "--objective",
        choices=["f1", "precision", "recall", "accuracy"],
        default="f1",
    )
    p_hill.add_argument("--out-dir", default="tmp/image_gating_hillclimb")
    p_hill.set_defaults(func=run_hillclimb)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "eval" and args.mode == "inline" and not args.prompt_file:
        parser.error("--prompt-file is required when --mode inline")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
