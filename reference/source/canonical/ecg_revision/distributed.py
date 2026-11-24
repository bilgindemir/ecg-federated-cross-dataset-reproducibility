"""Actual multi-host timing over persistent SSH streams (no fake latency).

Workers receive model parameters and retain their raw ECG shard locally.
Only application payload bytes are measured, not TCP/SSH wire overhead.
No machines are provisioned or paid resources started by this module.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import select
import shlex
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import Config, clean_json, clock, code_hash, environment, fingerprint, read_json, seed_all, write_json
from .data import Store
from .training import CNN, aggregate_states, local_update, save_torch, state_of

MAX_PAYLOAD = 64 * 1024**2

def encode_state(state):
    b = io.BytesIO(); np.savez(b, **{k:v.detach().cpu().numpy() for k,v in state.items()}); return b.getvalue()

def decode_state(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        return {k:torch.from_numpy(np.array(z[k], copy=True)) for k in z.files}

def send_frame(stream, metadata, payload=b""):
    metadata = dict(metadata, payload_length=len(payload))
    header = json.dumps(clean_json(metadata), sort_keys=True, allow_nan=False).encode()
    wire = struct.pack(">Q", len(header)) + header + payload
    # FileIO writes may be partial for large model payloads.
    view = memoryview(wire)
    while view:
        written = stream.write(view)
        if written is None or written <= 0: raise BrokenPipeError("Frame write failed")
        view = view[written:]
    stream.flush()
    return len(wire)

def read_exact(stream, length, timeout=None):
    data = bytearray(); end = time.monotonic()+timeout if timeout is not None else None
    while len(data) < length:
        if end is not None:
            remaining = end-time.monotonic()
            if remaining <= 0 or not select.select([stream], [], [], remaining)[0]: raise TimeoutError("Remote worker timeout")
            part = os.read(stream.fileno(), length-len(data))
        else: part = stream.read(length-len(data))
        if not part: raise EOFError("Worker stream ended; inspect its stderr log")
        data.extend(part)
    return bytes(data)

def receive_frame(stream, timeout=None):
    hlen = struct.unpack(">Q", read_exact(stream, 8, timeout))[0]
    if hlen > 1024**2: raise ValueError("Unexpected protocol header")
    metadata = json.loads(read_exact(stream, hlen, timeout))
    length = int(metadata["payload_length"])
    if not 0 <= length <= MAX_PAYLOAD: raise ValueError("Unexpected model payload length")
    payload = read_exact(stream, length, timeout)
    return metadata, payload, 8+hlen+length

def cloud_location():
    def get(url, headers=None, method="GET"):
        request = urllib.request.Request(url, headers=headers or {}, method=method)
        with urllib.request.urlopen(request, timeout=1) as response: return response.read().decode()
    try:
        zone = get("http://metadata.google.internal/computeMetadata/v1/instance/zone", {"Metadata-Flavor":"Google"}).rsplit("/",1)[-1]
        return dict(provider="gcp", region=zone.rsplit("-",1)[0], zone=zone, source="instance metadata")
    except Exception: pass
    try:
        token = get("http://169.254.169.254/latest/api/token", {"X-aws-ec2-metadata-token-ttl-seconds":"60"}, "PUT")
        document = json.loads(get("http://169.254.169.254/latest/dynamic/instance-identity/document", {"X-aws-ec2-metadata-token":token}))
        return dict(provider="aws", region=document["region"], zone=document["availabilityZone"], source="instance metadata")
    except Exception: pass
    try:
        document = json.loads(get("http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01", {"Metadata":"true"}))
        return dict(provider="azure", region=document["location"], source="instance metadata")
    except Exception: return dict(provider=None, region=None, source="not automatically verified")

def machine_info(check_cloud=True):
    identity = Path("/etc/machine-id")
    machine = identity.read_text().strip() if identity.exists() else socket.gethostname()
    return dict(hostname=socket.gethostname(), machine_id_sha256=hashlib.sha256(machine.encode()).hexdigest(),
                location=cloud_location() if check_cloud else {}, environment=environment())

def worker(settings_path, client_id):
    settings = read_json(settings_path); cfg = Config(**settings["experiment"]); cfg.validate()
    if cfg.split_policy != "patient42": raise ValueError("Timing study defaults to the clean cohort")
    started = clock()
    # Download/progress messages must not corrupt the binary stdout protocol.
    with contextlib.redirect_stdout(sys.stderr):
        store = Store(**settings["store"])
        if store.split_policy != cfg.split_policy: raise ValueError("Worker store/config split mismatch")
        store.prepare((f"client_{client_id}",))
        pack = store.pack(f"client_{client_id}", settings.get("fast_root"))
        view = pack.view(tau=cfg.train_tau)
    info = machine_info()
    send_frame(sys.stdout.buffer, dict(event="ready", client_id=client_id, n_eligible=len(view),
        data_id=store.data_id, code_hash=code_hash(), experiment_fingerprint=fingerprint(cfg.as_dict()), setup_s=clock()-started, **info))
    while True:
        request, payload, received = receive_frame(sys.stdin.buffer)
        if request["operation"] == "stop":
            send_frame(sys.stdout.buffer, {"event":"stopped"}); return
        if request["operation"] == "ping":
            send_frame(sys.stdout.buffer, {"event":"pong"}); continue
        if request["operation"] != "train": raise ValueError("Unknown worker operation")
        t0 = clock(); state = decode_state(payload)
        local_cfg = replace(cfg, seed=int(request["seed"]))
        updated, statistics, _ = local_update(state, view, local_cfg, int(request["round"]), client_id)
        encoded = encode_state(updated); worker_elapsed = clock()-t0
        send_frame(sys.stdout.buffer, dict(event="update", client_id=client_id, round=int(request["round"]),
            worker_compute_decode_encode_s=worker_elapsed, statistics=statistics,
            incoming_model_payload_bytes=len(payload), incoming_frame_bytes=received), encoded)

class Peer:
    def __init__(self, host, directory):
        self.host = host
        alias = host["ssh_alias"]
        if not re.fullmatch(r"[A-Za-z0-9_@.:-]+", alias): raise ValueError("Use a valid configured SSH alias")
        self.err = open(Path(directory) / f"client_{host['client_id']}_stderr.log", "ab")
        remote = shlex.join([host.get("python", "python3"), "-u", "-m", "ecg_revision.distributed", "worker",
                            "--settings", host["settings_path"], "--client", str(host["client_id"])])
        command = "cd " + shlex.quote(host["project_dir"]) + " && " + remote
        self.process = subprocess.Popen(["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", alias, command],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, bufsize=0)
        try:
            self.ready, _, _ = receive_frame(self.process.stdout, timeout=7200)
            if self.ready["event"] != "ready": raise ValueError("Worker not ready")
        except Exception:
            self.process.terminate(); self.err.close(); raise

    def exchange(self, metadata, payload=b""):
        started = time.perf_counter()
        sent = send_frame(self.process.stdin, metadata, payload)
        response, data, received = receive_frame(self.process.stdout, timeout=3600)
        cycle = time.perf_counter()-started
        return response, data, dict(exchange_wall_s=cycle, outbound_frame_bytes=sent, inbound_frame_bytes=received,
                                   outbound_model_payload_bytes=len(payload), inbound_model_payload_bytes=len(data))

    def close(self):
        try:
            if self.process.poll() is None:
                self.exchange({"operation":"stop"}); self.process.wait(timeout=10)
        except Exception:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.process.kill()
        self.err.close()

def coordinate(hosts_path, settings_path, output_dir, repeats=3, allow_declared_locations=False):
    settings = read_json(settings_path); cfg = Config(**settings["experiment"]); cfg.validate()
    hosts = read_json(hosts_path)
    if sorted(h["client_id"] for h in hosts) != [0,1,2,3]: raise ValueError("Exactly four distinct client IDs required")
    hosts = sorted(hosts, key=lambda h:h["client_id"])
    if len({h["ssh_alias"] for h in hosts}) != 4: raise ValueError("Four separate hosts required")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    peers, rows = [], []
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(Peer, h, output) for h in hosts]
            # Keep successfully started peers so exceptions do not orphan them.
            errors = []
            for future in futures:
                try: peers.append(future.result())
                except Exception as exc: errors.append(exc)
            if errors: raise errors[0]
        peers.sort(key=lambda p:p.host["client_id"])
        ready = [p.ready for p in peers]
        if len({r["machine_id_sha256"] for r in ready}) != 4: raise ValueError("Workers are not four distinct machines")
        if len({r["data_id"] for r in ready}) != 1 or any(r["code_hash"] != code_hash() for r in ready):
            raise ValueError("Code/cohort differs across coordinator and clients")
        if any(r["experiment_fingerprint"] != fingerprint(cfg.as_dict()) for r in ready):
            raise ValueError("Worker experiment settings differ from coordinator settings")
        verified = all(r["location"].get("region") for r in ready)
        regions = {r["location"].get("region") for r in ready} if verified else {h.get("declared_region") for h in hosts}
        if not verified and not allow_declared_locations:
            raise ValueError("Region metadata unavailable. Document physical locations; --allow-declared-locations explicitly labels evidence as declared, not verified.")
        if not verified and any(not r or "REPLACE" in str(r) for r in regions):
            raise ValueError("Replace all declared region placeholders with documented actual locations")
        if None in regions or len(regions)<2: raise ValueError("At least two geographically distinct regions required")
        topology = dict(coordinator=machine_info(), workers=ready, host_declarations=hosts,
            geographic_evidence="instance-metadata regions" if verified else "user-declared; external evidence required",
            experiment=cfg.as_dict(), repeats=repeats, transport="persistent SSH; raw ECG data stay on workers",
            bytes_definition="application model payload and framing; excludes TCP, SSH encryption and network-layer overhead")
        write_json(output / "topology.json", topology)
        latencies = []
        for peer in peers:
            for ping in range(10):
                _, _, times = peer.exchange({"operation":"ping"})
                latencies.append(dict(client_id=peer.host["client_id"], ping=ping, rtt_s=times["exchange_wall_s"]))
        pd.DataFrame(latencies).to_csv(output / "latency_rtt.csv", index=False)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for repetition in range(repeats):
                trial = replace(cfg, seed=cfg.seed+repetition)
                seed_all(trial.seed); state = state_of(CNN(trial.architecture)); moments=None
                # One untimed local-training warm-up per node; discard returned weights.
                warm = encode_state(state)
                list(pool.map(lambda p:p.exchange({"operation":"train", "round":0, "seed":trial.seed}, warm), peers))
                run_start = time.perf_counter()
                for round_number in range(1, trial.rounds+1):
                    round_start = time.perf_counter(); payload=encode_state(state)
                    futures = [pool.submit(p.exchange, {"operation":"train", "round":round_number, "seed":trial.seed}, payload) for p in peers]
                    updates, counts, clients = [], [], []
                    for peer, future in zip(peers, futures):
                        response, encoded, times = future.result()
                        if response["event"]!="update" or response["round"]!=round_number: raise ValueError("Unexpected update")
                        updates.append(decode_state(encoded)); counts.append(response["statistics"]["n_eligible"])
                        clients.append(dict(client_id=peer.host["client_id"], **response["statistics"], **times,
                            worker_compute_decode_encode_s=response["worker_compute_decode_encode_s"],
                            transport_dispatch_residual_s=times["exchange_wall_s"]-response["worker_compute_decode_encode_s"]))
                    aggregate_start=time.perf_counter()
                    state,moments,alpha=aggregate_states(updates,counts,trial.weighting,state,trial,moments)
                    ended=time.perf_counter()
                    rows.append(dict(repetition=repetition, seed=trial.seed, round=round_number,
                        round_wall_s=ended-round_start, aggregation_s=ended-aggregate_start,
                        elapsed_training_s=ended-run_start, clients=clients, alpha=alpha))
                    write_json(output / "rounds.json", rows)
                    print(f"distributed repeat={repetition+1}/{repeats} round={round_number}/{trial.rounds}: {ended-round_start:.2f}s", flush=True)
                save_torch(output / f"final_state_repeat_{repetition}.pt", dict(state=state, config=trial.as_dict(),
                    data_id=ready[0]["data_id"], origin="systems_experiment_final_round", best_round=None, seen_records=[]))
        flat = []
        for row in rows:
            for client in row["clients"]: flat.append(dict(**{k:v for k,v in row.items() if k not in ("clients","alpha")}, **client))
        pd.DataFrame(flat).to_csv(output / "client_round_timings.csv", index=False)
        totals=[]
        for repetition in range(repeats):
            selected=[r for r in rows if r["repetition"]==repetition]
            totals.append(dict(repetition=repetition,
                training_rounds_s=sum(r["round_wall_s"] for r in selected),
                end_to_end_training_loop_s=selected[-1]["elapsed_training_s"],
                model_payload_bytes=sum(c["outbound_model_payload_bytes"]+c["inbound_model_payload_bytes"] for r in selected for c in r["clients"]),
                examples=sum(c["examples"] for r in selected for c in r["clients"])))
        pd.DataFrame(totals).to_csv(output / "timing_totals.csv", index=False)
        write_json(output / "complete.json", dict(status="completed", repeats=repeats, rounds=cfg.rounds,
            geographic_evidence=topology["geographic_evidence"], excluded="setup, raw data download, warm-up, prediction evaluation; checkpoint/log I/O excluded from round_wall_s",
            residual_interpretation="Transport/dispatch residual is NOT isolated one-way latency; no clocks synchronized across nodes"))
        return pd.DataFrame(totals)
    finally:
        for peer in peers: peer.close()

def local_controls(settings_path, output_dir, repeats=3):
    """Measured sequential FL and pooled-work controls on the coordinator.

    Multiple-GPU distributed time versus one-device time is a resource contrast;
    it is not an algorithmic speedup or a time-to-equal-quality comparison.
    """
    settings=read_json(settings_path); cfg=Config(**settings["experiment"]); cfg.validate()
    store=Store(**settings["store"]); store.prepare(("train",)); pack=store.pack("train",settings.get("fast_root"))
    clients=[pack.view(pack.index.client.to_numpy()==c,tau=cfg.train_tau) for c in range(4)]
    pooled=pack.view(tau=cfg.train_tau); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for kind in ("sequential_fl", "pooled_work_control"):
        for repeat in range(repeats):
            trial=replace(cfg,seed=cfg.seed+repeat,method="pooled" if kind=="pooled_work_control" else cfg.method)
            seed_all(trial.seed); state=state_of(CNN(trial.architecture)); moments=None; central_optimizer=None
            views=[pooled] if kind=="pooled_work_control" else clients
            for c,view in enumerate(views): local_update(state,view,trial,0,c)
            start=clock(); examples=0
            for r in range(1,trial.rounds+1):
                updates=[]
                for c,view in enumerate(views):
                    if kind=="pooled_work_control":
                        updated,stats,_,central_optimizer=local_update(state,view,trial,r,c,optimizer_state=central_optimizer,return_optimizer=True)
                    else:updated,stats,_=local_update(state,view,trial,r,c)
                    updates.append(updated);examples+=stats["examples"]
                if len(updates)==1:state=updates[0]
                else:state,moments,_=aggregate_states(updates,[len(v) for v in views],trial.weighting,state,trial,moments)
            rows.append(dict(kind=kind,repetition=repeat,total_s=clock()-start,examples=examples))
            pd.DataFrame(rows).to_csv(out/"local_controls.csv",index=False)
    write_json(out/"local_control_environment.json",environment())
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    w=sub.add_parser("worker");w.add_argument("--settings",required=True);w.add_argument("--client",type=int,required=True)
    c=sub.add_parser("coordinate");c.add_argument("--settings",required=True);c.add_argument("--hosts",required=True);c.add_argument("--output",required=True);c.add_argument("--repeats",type=int,default=3);c.add_argument("--allow-declared-locations",action="store_true")
    l=sub.add_parser("local-controls");l.add_argument("--settings",required=True);l.add_argument("--output",required=True);l.add_argument("--repeats",type=int,default=3)
    a=parser.parse_args()
    if a.command=="worker":worker(a.settings,a.client)
    elif a.command=="coordinate":coordinate(a.hosts,a.settings,a.output,a.repeats,a.allow_declared_locations)
    else:local_controls(a.settings,a.output,a.repeats)

if __name__=="__main__": main()
