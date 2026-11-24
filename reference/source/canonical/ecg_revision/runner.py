"""Notebook action entry point; runs in an isolated Colab virtual environment."""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
import pandas as pd
from .common import Config, code_hash, environment, read_json, write_json
from .data import Store
from .audit import audit_legacy_drive
from .training import run
from .studies import tune, reporting_grid, save_plan, run_jobs, baselines, collect
from .evaluation import evaluate_d2, evaluate_dense, benchmark_dense
from .privacy import membership_audit

def completed_checkpoints(root, store, baseline_settings_only=False, include_baselines=True):
    found=[]
    for p in sorted((Path(root)/"runs").glob("*/complete.json")):
        done=read_json(p);cfg=done["config"]
        if done["data_id"]!=store.data_id or done["phase"]!="report":continue
        if baseline_settings_only and (cfg["local_epochs"]!=1 or cfg["weighting"]!="equal"):continue
        if not include_baselines and cfg["method"] not in ("fedavg","fedavgm","fedprox","fedadam"):continue
        found.append(p.parent/"best.pt")
    if not found:raise ValueError("No completed report checkpoints. Run tune/main (and baselines if needed) first.")
    return found

def frozen_selection(root, base, store):
    p=Path(root)/"selected_hyperparameters.json"
    if not p.exists():raise ValueError("Run ACTION='tune' first. Report settings are selected using validation only.")
    selected=read_json(p)
    if selected["data_id"]!=store.data_id or selected["code_hash"]!=code_hash():raise ValueError("Frozen tuning provenance differs. Use the original code or a new study folder.")
    for key in ("rounds","batch_size","architecture","schedule","imbalance","selection","train_tau","split_policy","unknown_policy","lr"):
        if selected["base_config"][key]!=base.as_dict()[key]:raise ValueError(f"Frozen tuning setting changed: {key}")
    return selected

def export_tables(base_dir, study_root, store):
    base_dir,study_root=Path(base_dir),Path(study_root)
    if list((study_root/"evaluations").glob("*/metrics_*.csv")):collect(study_root)
    target=base_dir/"ECG_FL_results_for_review.zip"
    temporary=target.with_suffix(".partial")
    with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for folder,label in [(study_root,"study"),(base_dir/"audit","audit"),(base_dir/"timing","timing"),(base_dir/"legacy_diagnostics","legacy_diagnostics"),(store.manifest_dir,"data_manifest")]:
            if not folder.exists():continue
            for p in sorted(folder.rglob("*")):
                if p.is_file() and p.suffix in (".json",".csv",".md",".txt"):
                    z.write(p,Path(label)/p.relative_to(folder))
        for name in ("environment.json","last_invocation.json","data_counts.csv"):
            p=base_dir/name
            if p.exists():z.write(p,Path("run_context")/name)
        z.writestr("READ_FIRST.txt","Only executed outputs are included. Check phases, coverage and actual seed counts. Absence of an output means it has not been completed. Raw ECGs and large prediction/checkpoint files are intentionally excluded.\n")
    temporary.replace(target)
    print(f"Review package: {target} ({target.stat().st_size/1024**2:.1f} MiB)")
    return target

def execute(settings):
    action=settings["action"]
    if action=="prepare_pilot":
        for step in ("self_test", "audit", "prepare", "pilot", "export"):
            print(f"Starting: {step}", flush=True)
            execute(dict(settings, action=step))
            print(f"Completed: {step}", flush=True)
        return
    base_dir=Path(settings["base_dir"]);root=base_dir/"study";root.mkdir(parents=True,exist_ok=True)
    fast=settings.get("fast_root")
    base=Config(**settings.get("experiment",{}));base.validate()
    store=Store(base_dir/"data",legacy_root=settings.get("legacy_root"),split_policy=base.split_policy,unknown_policy=base.unknown_policy)
    write_json(base_dir/"last_invocation.json",settings)
    write_json(base_dir/"environment.json",environment())
    if action=="self_test":
        # Unit/self-tests validate experiment logic, not GPU throughput.
        # Run them on CPU so CUDA/cuDNN determinism differences do not make
        # the preflight test hardware-dependent. Pilot/main still use the GPU.
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run([sys.executable, "-m", "pytest", str(Path(__file__).parent.parent / "tests"), "-q"],
                       check=True, env=env)
        return
    if action=="audit":
        report=audit_legacy_drive(settings.get("legacy_root", ""),store,base_dir/"audit")
        print("New split:",read_json(store.manifest_dir/"split_audit.json"))
        print("Old checkpoint candidates:",len([r for r in report["inventory"] if "architecture" in r]))
        print("Tau-output candidates:",report["tau_candidates"])
        print("Label cache checks:",report["ptb_primary_label_checks"])
        print("Full audit:",base_dir/"audit"/"legacy_audit.json");return
    if action=="prepare":
        print("Preparing public waveforms and D2 disk arrays. Existing complete files are reused.",flush=True)
        print("Temporary disk free GiB:",round(shutil.disk_usage("/content" if Path("/content").exists() else ".").free/1024**3,1))
        store.prepare();rows=[]
        for scope in ("train","val","test"):
            pack=store.pack(scope,fast)
            for domain,frame in pack.index.groupby("domain"):
                rows.append(dict(split=scope,domain=domain,windows=len(frame),patients=frame.patient.nunique(),
                    records=frame.rid.nunique(),positive=int(frame.y.sum()),negative=int((frame.y==0).sum())))
        pd.DataFrame(rows).to_csv(base_dir/"data_counts.csv",index=False)
        print(pd.DataFrame(rows).to_string(index=False));return
    if action=="pilot":
        jobs=[replace(base,phase="pilot",method=m,rounds=2,seed=909) for m in ("fedavg","fedavgm","fedprox","fedadam")]
        run_jobs(jobs,store,root,fast);return
    if action=="tune":
        selected=tune(store,root,base,fast,joint_lr_grid=settings.get("joint_lr_grid",False))
        print(selected);return
    seeds=tuple(settings.get("report_seeds",[0,1,2,3,4]))
    if action=="main":
        selected=frozen_selection(root,base,store)
        jobs=reporting_grid(selected,base,seeds,settings.get("include_interaction",False))
        plan=save_plan(jobs,store,root)
        print(f"{len(jobs)} jobs; {int(plan.expected_examples.sum()):,} planned training examples",flush=True)
        run_jobs(jobs,store,root,fast);return
    if action=="baselines":
        frozen_selection(root,base,store)
        baselines(store,root,base,seeds,fast);return
    if action=="evaluate":
        for checkpoint in completed_checkpoints(root,store):
            evaluate_d2(checkpoint,store,root,bootstraps=settings.get("bootstraps",1000),fast_root=fast)
        print(collect(root).to_string(index=False));return
    if action=="dense":
        checkpoints=completed_checkpoints(root,store,True,False)
        print(benchmark_dense(checkpoints[0],store,root,batch_size=base.batch_size).to_string(index=False),flush=True)
        cap=None if settings.get("dense_mode","full")=="full" else int(settings.get("dense_cap",16384))
        for checkpoint in checkpoints:
            evaluate_d2(checkpoint,store,root,bootstraps=settings.get("bootstraps",1000),fast_root=fast)
            evaluate_dense(checkpoint,store,root,cap_per_incart_record=cap,bootstraps=settings.get("bootstraps",1000),batch_size=base.batch_size)
        collect(root);return
    if action=="privacy":
        for checkpoint in completed_checkpoints(root,store,True,True):
            from .training import load_checkpoint
            if load_checkpoint(checkpoint)["config"]["method"] not in ("fedavg","fedavgm","fedprox","fedadam","pooled"):continue
            print(membership_audit(checkpoint,store,root,fast,settings.get("bootstraps",1000)).to_string(index=False))
        return
    if action=="export":export_tables(base_dir,root,store);return
    raise ValueError("Unknown ACTION")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--settings",required=True);args=parser.parse_args()
    execute(read_json(args.settings))
