"""Run the canonical configurations with their fingerprint-matched source."""

import argparse
import csv
from dataclasses import fields
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "train", "evaluate"))
    parser.add_argument("--root", default="work/frozen_canonical", type=Path)
    parser.add_argument("--seed", type=int, choices=range(10))
    parser.add_argument("--reference-checkpoints", action="store_true",
                        help="Score the published validation-selected checkpoints on D2")
    args = parser.parse_args()
    if args.reference_checkpoints and args.action != "evaluate":
        parser.error("--reference-checkpoints applies to evaluate")

    reference = Path(__file__).resolve().parents[1] / "reference"
    sys.path.insert(0, str(reference / "source" / "canonical"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from ecg_revision.common import Config, code_hash, file_hash
    from ecg_revision.data import Store
    from ecg_revision.evaluation import evaluate_d2
    from ecg_revision.training import run, run_id

    torch.set_num_threads(1)
    study = args.root.resolve() / "study"
    protocol = json.loads((reference / "study" / "primary_canonical_protocol.json").read_text())
    if code_hash() != protocol["code_hash"]:
        raise SystemExit("Frozen canonical source fingerprint does not match the recorded protocol.")

    store = None
    if args.action != "plan":
        store = Store(study.parent / "data", split_policy="patient42", unknown_policy="exclude")
        if store.data_id != protocol["data_id"]:
            raise SystemExit("Record manifest fingerprint does not match the canonical protocol.")

    hashes = {}
    if args.reference_checkpoints:
        with (reference / "canonical_checkpoints.csv").open(newline="") as stream:
            hashes = {r["run_id"]: r["checkpoint_sha256"] for r in csv.DictReader(stream)}
    with (reference / "study" / "primary_canonical_runs.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if args.seed is not None and int(row["seed"]) != args.seed:
                continue
            recorded_id = Path(row["run_path"]).name
            saved = json.loads((reference / "runs" / recorded_id / "config.json").read_text())
            config = Config(**{field.name: saved[field.name] for field in fields(Config)})
            if run_id(config, protocol["data_id"]) != recorded_id:
                raise SystemExit(f"Run fingerprint mismatch for {config.method}, seed {config.seed}.")
            if args.action == "plan":
                print(config.method, config.seed, recorded_id)
            elif args.action == "train":
                run(config, store, study)
            else:
                source = reference if args.reference_checkpoints else study
                checkpoint = source / "runs" / recorded_id / "best.pt"
                if args.reference_checkpoints and file_hash(checkpoint) != hashes[recorded_id]:
                    raise SystemExit(f"Reference checkpoint SHA-256 mismatch for {recorded_id}.")
                if not checkpoint.is_file():
                    raise SystemExit(f"Train the canonical run before evaluating {checkpoint}.")
                evaluate_d2(checkpoint, store, study, taus=(0.05,))


if __name__ == "__main__":
    main()
