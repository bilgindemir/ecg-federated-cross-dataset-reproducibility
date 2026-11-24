from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from .common import Config, code_hash, fingerprint, read_json, write_json
from .training import run

METHODS = ("fedavg", "fedavgm", "fedprox", "fedadam")

def tuning_grid(base=Config(), joint_lr_grid=False, seeds=(101, 202, 303)):
    """Optimizer-specific validation-only tuning under a matched protocol.

    FedProx includes mu=0 as the FedAvg limiting case and extends beyond the
    previously tested upper range. FedAdam and FedAvgM use the same server
    learning-rate grid, including 1.0 so the historical implicit FedAvgM
    server step is represented. If joint_lr_grid=True, the same client-LR
    grid is crossed with every optimizer-specific candidate.
    """
    jobs = []
    client_lrs = (.0003, .001, .003) if joint_lr_grid else (base.lr,)

    grids = {
        "fedprox": (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        "fedadam": (1e-3, 1e-2, 1e-1, 1.0),
        "fedavgm": (1e-3, 1e-2, 1e-1, 1.0),
    }

    for method, values in grids.items():
        for value, lr, seed in itertools.product(values, client_lrs, seeds):
            jobs.append(
                replace(
                    base,
                    method=method,
                    phase="tuning",
                    seed=seed,
                    lr=lr,
                    mu=value if method == "fedprox" else base.mu,
                    server_lr=value if method in ("fedadam", "fedavgm") else base.server_lr,
                    local_epochs=1,
                    weighting="equal",
                )
            )
    return jobs


def tune(store, output_root, base=Config(), fast_root=None, joint_lr_grid=False, seeds=(101, 202, 303)):
    jobs = tuning_grid(base, joint_lr_grid, seeds)
    grid_id = fingerprint(
        dict(
            jobs=[j.canonical() for j in jobs],
            data_id=store.data_id,
            code_hash=code_hash(),
        )
    )
    selected_path = Path(output_root) / "selected_hyperparameters.json"

    if selected_path.exists():
        old = read_json(selected_path)
        if old["grid_id"] != grid_id:
            raise ValueError(
                "Frozen grid changed; use a new output root for a new preregistered analysis"
            )
        return old

    rows = []
    for cfg in jobs:
        path = run(cfg, store, output_root, fast_root)
        done = read_json(path / "complete.json")
        rows.append(
            dict(
                **cfg.as_dict(),
                best_validation=done["best_score"],
                run_path=str(path),
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_root) / "tuning_all_runs.csv", index=False)

    candidates = {
        "fedprox": 6,
        "fedadam": 4,
        "fedavgm": 4,
    }
    selected = dict(
        grid_id=grid_id,
        data_id=store.data_id,
        code_hash=code_hash(),
        tuning_seeds=list(seeds),
        selection="mean of best validation macro-domain AUC across tuning seeds",
        checkpoint_selection="best round by validation macro-domain AUC within each run",
        test_data_used=False,
        candidate_count_per_method=candidates,
        joint_client_lr_grid=bool(joint_lr_grid),
        fixed_rounds=base.rounds,
        base_config=base.as_dict(),
    )

    for method, parameter in (
        ("fedprox", "mu"),
        ("fedadam", "server_lr"),
        ("fedavgm", "server_lr"),
    ):
        group = (
            df[df.method == method]
            .groupby([parameter, "lr"])
            .best_validation
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        if not (group["count"] == len(seeds)).all():
            raise ValueError(f"Incomplete tuning grid for {method}")

        # Prespecified deterministic tie-break: highest mean validation AUC,
        # then the lower optimizer-specific coefficient, then lower client LR.
        group = group.sort_values(
            ["mean", parameter, "lr"],
            ascending=[False, True, True],
        )
        group.to_csv(
            Path(output_root) / f"tuning_{method}_summary.csv",
            index=False,
        )

        best = group.iloc[0]
        selected[method] = {
            parameter: float(best[parameter]),
            "lr": float(best["lr"]),
            "mean_validation": float(best["mean"]),
        }

    write_json(selected_path, selected)
    return selected


def apply_tuning(cfg, selected):
    if cfg.method not in ("fedprox", "fedadam", "fedavgm"):
        return cfg
    values = {
        k: v
        for k, v in selected[cfg.method].items()
        if k in ("mu", "lr", "server_lr")
    }
    return replace(cfg, **values)


def reporting_grid(selected, base=Config(), seeds=(0, 1, 2, 3, 4), interaction=False):
    # Minimal design: equal-weight E=1/3/5; data-weight E=1.
    # No weight-by-epoch interaction claim without interaction=True.
    settings = [("equal", 1), ("samples", 1), ("equal", 3), ("equal", 5)]
    if interaction: settings += [("samples", 3), ("samples", 5)]
    jobs = []
    for method, seed, (weighting, epochs) in itertools.product(METHODS, seeds, settings):
        cfg = replace(base, method=method, phase="report", seed=seed, weighting=weighting, local_epochs=epochs)
        jobs.append(apply_tuning(cfg, selected))
    return jobs

def save_plan(jobs, store, output_root):
    train = store.pack("train"); sizes = train.index.groupby("client").size().to_dict()
    rows = []
    for cfg in jobs:
        examples = cfg.rounds*cfg.local_epochs*(sum(sizes.values()) if cfg.schedule != "fixed_steps" else 4*cfg.steps_per_epoch*cfg.batch_size)
        rows.append(dict(**cfg.as_dict(), expected_examples=examples,
            expected_local_updates=cfg.rounds*cfg.local_epochs*sum(int(np.ceil(n/cfg.batch_size)) for n in sizes.values())
                if cfg.schedule != "fixed_steps" else cfg.rounds*cfg.local_epochs*4*cfg.steps_per_epoch))
    frame = pd.DataFrame(rows); frame.to_csv(Path(output_root) / "execution_plan.csv", index=False)
    write_json(Path(output_root) / "execution_plan.json", dict(data_id=store.data_id, code_hash=code_hash(), jobs=rows,
        job_count=len(jobs), total_expected_examples=int(frame.expected_examples.sum()),
        note="Compute estimates are work counts, not a promised runtime; E is n_k draws per client in sampled_epoch mode."))
    return frame

def run_jobs(jobs, store, output_root, fast_root=None):
    paths = []
    for cfg in jobs:
        try: paths.append(str(run(cfg, store, output_root, fast_root)))
        except Exception as exc:
            write_json(Path(output_root) / "last_failure.json", dict(config=cfg.as_dict(), error=type(exc).__name__, message=str(exc)))
            raise
    write_json(Path(output_root) / "last_completed_jobs.json", paths)
    return paths

def baselines(store, output_root, base=Config(), seeds=(0, 1, 2, 3, 4), fast_root=None):
    """PTB-only (zero-shot on INCART), INCART-only, pooled, and PTB->INCART.

    Same architecture. No source test records in pretraining. The pooled model
    gets R passes over all training windows. Transfer gets R source passes and
    R target passes: total sampled examples match pooled when E=1. Updates and
    elapsed compute are logged; these are not identical optimization paths.
    """
    output = []
    for seed in seeds:
        source_cfg = replace(base, method="ptb_only", seed=seed, phase="report", local_epochs=1, selection="ptb_auc")
        source = run(source_cfg, store, output_root, fast_root); output.append(str(source))
        for method, selection in [("incart_only", "incart_auc"), ("pooled", "macro_domain_auc")]:
            cfg = replace(base, method=method, seed=seed, phase="report", local_epochs=1, selection=selection)
            output.append(str(run(cfg, store, output_root, fast_root)))
        transfer_cfg = replace(base, method="transfer", seed=seed, phase="report", local_epochs=1, selection="incart_auc")
        output.append(str(run(transfer_cfg, store, output_root, fast_root, initial_checkpoint=source / "best.pt")))
    write_json(Path(output_root) / "baseline_jobs.json", output)
    return output

def collect(output_root):
    """No five-seed label unless five distinct seeds actually exist."""
    root = Path(output_root); frames = []
    for p in sorted((root / "evaluations").glob("*/metrics_*.csv")):
        frame = pd.read_csv(p); frame["evaluation_file"] = str(p); frames.append(frame)
    if not frames: raise ValueError("No completed evaluation tables")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(root / "all_evaluation_rows.csv", index=False)
    fields = ["data_id", "origin", "phase", "method", "weighting", "local_epochs", "schedule", "imbalance", "selection", "train_tau", "architecture", "rounds", "batch_size", "lr", "mu", "server_lr", "representation", "eval_tau", "domain"]
    from .metrics import METRICS
    # Origins/phases prevent legacy, pilot and main results from being mixed.
    grouped = df.groupby(fields, dropna=False)
    rows = []
    for keys, group in grouped:
        if group.seed.duplicated().any() and len(group)>1: raise ValueError("Duplicate seed/evaluation configurations: inspect all_evaluation_rows.csv")
        row = dict(zip(fields, keys)); row.update(n_seeds=group.seed.nunique(), n_runs=len(group))
        for metric in METRICS:
            row[metric+"_mean"] = group[metric].mean(); row[metric+"_sd"] = group[metric].std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows); summary.to_csv(root / "summary_across_seeds.csv", index=False)
    # Paired method differences within the same seed/configuration.
    context = [f for f in fields if f not in ("method", "mu", "server_lr", "lr")] + ["seed"]
    paired = []
    for keys, group in df.groupby(context, dropna=False):
        methods = group.set_index("method")
        if methods.index.duplicated().any(): raise ValueError("Ambiguous paired contrast: multiple hyperparameter settings for a method. Use a separate output root per frozen study.")
        if "fedprox" not in methods.index: continue
        for comparator in ("fedavg", "fedavgm", "fedadam"):
            if comparator not in methods.index: continue
            row = dict(zip(context, keys)); row["contrast"] = "fedprox-minus-"+comparator
            for metric in METRICS: row[metric+"_delta"] = methods.loc["fedprox", metric]-methods.loc[comparator, metric]
            paired.append(row)
    pd.DataFrame(paired).to_csv(root / "paired_seed_differences.csv", index=False)
    return summary
