#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-Stage SFT with VERL FSDP + FlashAttention-2 (H100/H200) - r4
- NEW: --max_steps to hard-cap training steps (e.g., 200).
- Enforces one epoch = max_steps by limiting data.train_max_samples.
- Keeps r3 safety: Torch>=2.4 -> use model.strategy=fsdp and disable grad clipping.

Other retained fixes:
- Remove failing Hydra override for checkpoint struct.
- Rebuild train.parquet if empty; abort if still empty.
"""

from __future__ import annotations
import os, sys, json, math, time, argparse, datetime, subprocess, signal, threading, io, gzip, bz2, gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from contextlib import contextmanager

# -------------------- tiny logging --------------------
def _now() -> str: 
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str) -> None: 
    print(f"[{_now()}] {msg}", flush=True)

# -------------------- quick req checks --------------------
def _require(mod: str, hint: str):
    try: 
        __import__(mod)
        log(f"✓ Module '{mod}' available")
    except Exception as e:
        log(f"✗ Missing '{mod}'. Install it. Hint: {hint}")
        raise SystemError(f"Missing '{mod}'. Install it. Hint: {hint}\n{e}")

# -------------------- config --------------------
@dataclass
class StageCfg:
    num: int
    max_len: int
    hours: float
    lr: float
    warmup_ratio: float
    micro_bsz: int
    grad_accum: int
    save_freq: int

@dataclass
class RunCfg:
    # IO
    train_file: Path
    eval_file: Path
    output_dir: Path
    base_model: str

    # time & steps
    stage1_hours: float
    stage2_hours: float
    stage3_hours: float
    est_step_seconds: float
    max_steps: Optional[int]  # <-- NEW: hard cap

    # LoRA
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: str

    # LR, warmup, batch
    stage1_lr: float; stage1_warmup: float; stage1_micro_bsz: int; stage1_grad_accum: int
    stage2_lr: float; stage2_warmup: float; stage2_micro_bsz: int; stage2_grad_accum: int
    stage3_lr: float; stage3_warmup: float; stage3_micro_bsz: int; stage3_grad_accum: int

    # context
    len1: int; len2: int; len3: int

    # eval
    per_step_eval_samples: int
    final_eval_samples: int
    gen_samples: int
    gen_max_new: int

    # buckets
    use_buckets: bool
    chars_per_token: float
    bucket_margin: float

    # misc
    seed: int
    save_freq1: int
    save_freq2: int
    save_freq3: int

# -------------------- args --------------------
def _parse_args() -> RunCfg:
    ap = argparse.ArgumentParser("3-Stage VERL SFT (H100/H200) with FA2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--train_file", type=Path, required=True)
    ap.add_argument("--eval_file",  type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    # time budget (defaults sum to 14h)
    ap.add_argument("--stage1_hours", type=float, default=4.0)
    ap.add_argument("--stage2_hours", type=float, default=5.0)
    ap.add_argument("--stage3_hours", type=float, default=5.0)
    ap.add_argument("--est_step_seconds", type=float, default=22.0)

    # NEW: hard cap steps
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Hard cap on training steps per stage (overrides time/epochs logic)")

    # LoRA
    ap.add_argument("--lora_rank", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=256)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", type=str,
                    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # stage hparams
    ap.add_argument("--stage1_lr", type=float, default=1.5e-4)
    ap.add_argument("--stage1_warmup", type=float, default=0.08)
    ap.add_argument("--stage1_micro_bsz", type=int, default=8)
    ap.add_argument("--stage1_grad_accum", type=int, default=4)

    ap.add_argument("--stage2_lr", type=float, default=8.0e-5)
    ap.add_argument("--stage2_warmup", type=float, default=0.05)
    ap.add_argument("--stage2_micro_bsz", type=int, default=4)
    ap.add_argument("--stage2_grad_accum", type=int, default=8)

    ap.add_argument("--stage3_lr", type=float, default=5.0e-5)
    ap.add_argument("--stage3_warmup", type=float, default=0.03)
    ap.add_argument("--stage3_micro_bsz", type=int, default=2)
    ap.add_argument("--stage3_grad_accum", type=int, default=16)

    # context
    ap.add_argument("--len1", type=int, default=2048)
    ap.add_argument("--len2", type=int, default=4096)
    ap.add_argument("--len3", type=int, default=8192)

    # eval & gen
    ap.add_argument("--per_step_eval_samples", type=int, default=1500)
    ap.add_argument("--final_eval_samples", type=int, default=6000)
    ap.add_argument("--gen_samples", type=int, default=128)
    ap.add_argument("--gen_max_new", type=int, default=256)

    # buckets
    ap.add_argument("--use_buckets", action="store_true", default=False)
    ap.add_argument("--chars_per_token", type=float, default=4.0)
    ap.add_argument("--bucket_margin", type=float, default=0.25)

    # save frequency
    ap.add_argument("--save_freq1", type=int, default=250)
    ap.add_argument("--save_freq2", type=int, default=250)
    ap.add_argument("--save_freq3", type=int, default=250)

    # misc
    ap.add_argument("--seed", type=int, default=42)

    ns = ap.parse_args()
    total = ns.stage1_hours + ns.stage2_hours + ns.stage3_hours
    if total > 14.5 and ns.max_steps is None:
        log(f"⚠ Warning: Total hours {total:.2f} exceeds 14.0. Proceeding anyway...")
    
    return RunCfg(
        train_file=ns.train_file, eval_file=ns.eval_file, output_dir=ns.output_dir,
        base_model=ns.base_model,
        stage1_hours=ns.stage1_hours, stage2_hours=ns.stage2_hours, stage3_hours=ns.stage3_hours,
        est_step_seconds=ns.est_step_seconds, max_steps=ns.max_steps,
        lora_rank=ns.lora_rank, lora_alpha=ns.lora_alpha, lora_dropout=ns.lora_dropout,
        lora_targets=ns.lora_targets,
        stage1_lr=ns.stage1_lr, stage1_warmup=ns.stage1_warmup, stage1_micro_bsz=ns.stage1_micro_bsz, stage1_grad_accum=ns.stage1_grad_accum,
        stage2_lr=ns.stage2_lr, stage2_warmup=ns.stage2_warmup, stage2_micro_bsz=ns.stage2_micro_bsz, stage2_grad_accum=ns.stage2_grad_accum,
        stage3_lr=ns.stage3_lr, stage3_warmup=ns.stage3_warmup, stage3_micro_bsz=ns.stage3_micro_bsz, stage3_grad_accum=ns.stage3_grad_accum,
        len1=ns.len1, len2=ns.len2, len3=ns.len3,
        per_step_eval_samples=ns.per_step_eval_samples, final_eval_samples=ns.final_eval_samples,
        gen_samples=ns.gen_samples, gen_max_new=ns.gen_max_new,
        use_buckets=ns.use_buckets, chars_per_token=ns.chars_per_token, bucket_margin=ns.bucket_margin,
        seed=ns.seed,
        save_freq1=ns.save_freq1, save_freq2=ns.save_freq2, save_freq3=ns.save_freq3,
    )

# -------------------- env --------------------
def _setup_env() -> Tuple[dict, int]:
    import torch
    ngpu = torch.cuda.device_count()
    env = os.environ.copy()

    # FA2 everywhere
    env["HF_ATTENTION_BACKEND"] = "flash_attention_2"
    env["TRANSFORMERS_ATTENTION_BACKEND"] = "flash_attention_2"
    env["TRANSFORMERS_USE_FLASH_ATTENTION_2"] = "1"
    env["USE_FLASH_ATTENTION_2"] = "1"
    env["FLASH_ATTENTION_SKIP_INPUT_CHECKS"] = "1"

    # CUDA / NCCL (safe defaults for single-node)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,expandable_segments:True"
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "64"
    env["NCCL_DEBUG"] = "WARN"
    env["NCCL_IB_DISABLE"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = env.get("MASTER_PORT", "29500")

    # tokenizers / workers
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "16"

    log(f"Environment configured for {ngpu} GPU(s)")
    return env, ngpu

# -------------------- parquet conversion --------------------
@contextmanager
def _open_textmaybe(path: Path):
    f = raw = reader = None
    try:
        suf = path.suffix.lower()
        if suf == ".gz":
            f = gzip.open(path, "rt", encoding="utf-8", errors="ignore")
        elif suf == ".bz2":
            f = bz2.open(path, "rt", encoding="utf-8", errors="ignore")
        elif suf == ".zst":
            try:
                import zstandard as zstd
                raw = open(path, "rb")
                dctx = zstd.ZstdDecompressor()
                reader = dctx.stream_reader(raw)
                f = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            except ImportError:
                log("⚠ zstandard not installed, treating .zst as plain text")
                f = open(path, "r", encoding="utf-8", errors="ignore")
        else:
            f = open(path, "r", encoding="utf-8", errors="ignore")
        yield f
    finally:
        for h in (f, reader, raw):
            try:
                if h: h.close()
            except: pass

def _convert_jsonl_to_parquet(src: Path, dst: Path) -> int:
    import pyarrow as pa, pyarrow.parquet as pq
    log(f"Converting {src} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    msg_struct = pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
    messages_type = pa.list_(msg_struct)
    schema = pa.schema([pa.field("messages", messages_type), pa.field("char_len", pa.int32())])
    writer = pq.ParquetWriter(str(dst), schema=schema, compression="zstd")

    n = 0
    buf_msgs, buf_len = [], []
    
    with _open_textmaybe(src) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try: 
                ex = json.loads(line)
            except json.JSONDecodeError:
                if line_num % 1000 == 0:
                    log(f"⚠ Skipping invalid JSON at line {line_num}")
                continue
                
            msgs = ex.get("messages") or ex.get("conversations")
            if not isinstance(msgs, list): 
                continue

            md = ex.get("metadata")
            if md is not None and msgs and msgs[0].get("role") != "system":
                if not isinstance(md, str):
                    md = json.dumps(md, ensure_ascii=False)
                msgs = [{"role": "system", "content": f"[METADATA]\n{md}"}] + msgs

            clean = []
            clen = 0
            for m in msgs:
                if not isinstance(m, dict): continue
                role = (m.get("role") or "").strip()
                if not role: continue
                content = m.get("content", "")
                if isinstance(content, (list, dict)):
                    content = json.dumps(content, ensure_ascii=False)
                s = str(content)
                clen += len(s)
                clean.append({"role": role, "content": s})
            
            if not clean: continue

            buf_msgs.append(clean)
            buf_len.append(int(clen))
            n += 1
            
            if len(buf_msgs) >= 10000:
                writer.write_table(pa.Table.from_arrays(
                    [pa.array(buf_msgs, type=messages_type), pa.array(buf_len, type=pa.int32())], 
                    schema=schema))
                buf_msgs.clear()
                buf_len.clear()
                
    if buf_msgs:
        writer.write_table(pa.Table.from_arrays(
            [pa.array(buf_msgs, type=messages_type), pa.array(buf_len, type=pa.int32())], 
            schema=schema))
    
    writer.close()
    log(f"✓ Converted {n:,} rows from {src.name}")
    return n

def _parquet_num_rows(path: Path) -> int:
    import pyarrow.parquet as pq
    if not path.exists(): return 0
    try:
        return pq.ParquetFile(str(path)).metadata.num_rows
    except Exception as e:
        log(f"⚠ Error reading parquet {path}: {e}")
        return 0

def _build_length_buckets(src: Path, cfg: RunCfg, data_root: Path) -> Dict[int, Path]:
    import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
    
    log(f"Building length buckets from {src.name}")
    pf = pq.ParquetFile(str(src))
    schema = pf.schema_arrow
    
    stage_paths = {
        1: data_root / "train_stage1.parquet",
        2: data_root / "train_stage2.parquet",
        3: data_root / "train_stage3.parquet",
    }
    for p in stage_paths.values():
        if p.exists(): p.unlink()

    writers = {k: pq.ParquetWriter(str(v), schema=schema, compression="zstd") for k, v in stage_paths.items()}
    counts = {1: 0, 2: 0, 3: 0}

    thr2 = int(cfg.bucket_margin * cfg.chars_per_token * cfg.len2)
    thr3 = int(cfg.bucket_margin * cfg.chars_per_token * cfg.len3)
    log(f"Bucket thresholds (chars): S1 < {thr2} | S2 [{thr2},{thr3}) | S3 >= {thr3}")

    try:
        for i in range(pf.num_row_groups):
            tbl = pf.read_row_group(i, columns=["messages", "char_len"])
            s1 = tbl.filter(pc.less(tbl["char_len"], pa.scalar(thr2)))
            s2 = tbl.filter(pc.and_(pc.greater_equal(tbl["char_len"], pa.scalar(thr2)),
                                    pc.less(tbl["char_len"], pa.scalar(thr3))))
            s3 = tbl.filter(pc.greater_equal(tbl["char_len"], pa.scalar(thr3)))
            if s1.num_rows: writers[1].write_table(s1); counts[1] += s1.num_rows
            if s2.num_rows: writers[2].write_table(s2); counts[2] += s2.num_rows
            if s3.num_rows: writers[3].write_table(s3); counts[3] += s3.num_rows
    finally:
        for w in writers.values(): w.close()
    
    log(f"✓ Bucket counts: S1={counts[1]:,}  S2={counts[2]:,}  S3={counts[3]:,}")
    return stage_paths

# -------------------- training & merging --------------------
def _calc_steps(hours: float, est_step_seconds: float) -> int:
    safe = max(0, int(hours * 3600) - 300)
    steps = max(50, safe // max(1, int(est_step_seconds)))
    return steps

def _torch_ge_24() -> bool:
    import torch
    try:
        major, minor = map(int, torch.__version__.split("+")[0].split(".")[:2])
        return (major, minor) >= (2, 4)
    except Exception:
        return False

def _build_overrides(
    cfg: RunCfg, ngpu: int, train_pq: Path, val_pq: Path, train_rows: int,
    stage: StageCfg, stage_root: Path, base_for_stage: str
) -> List[str]:
    global_bsz = stage.micro_bsz * stage.grad_accum * max(1, ngpu)
    if train_rows <= 0:
        raise SystemError("Train parquet has 0 rows; cannot compute steps/epoch. Check input JSONL content.")

    # Default epoch/steps
    steps_per_epoch = max(1, math.ceil(train_rows / global_bsz))
    target_steps = _calc_steps(stage.hours, cfg.est_step_seconds)
    epochs = max(1, math.ceil(target_steps / steps_per_epoch))
    total_steps = steps_per_epoch * epochs

    # Hard cap mode: override everything to exactly max_steps
    train_max_samples_override = None
    if cfg.max_steps is not None and cfg.max_steps > 0:
        total_steps = cfg.max_steps
        epochs = 1
        steps_per_epoch = cfg.max_steps
        train_max_samples_override = cfg.max_steps * global_bsz
        log(f"🛑 Hard step cap enabled: max_steps={cfg.max_steps} → "
            f"train_max_samples={train_max_samples_override}, epochs=1")

    val_max = cfg.per_step_eval_samples
    lora_targets = "[" + ",".join([f"'{t.strip()}'" for t in cfg.lora_targets.split(",") if t.strip()]) + "]"

    log(f"Stage {stage.num} config:")
    log(f"  - Context: {stage.max_len} tokens")
    log(f"  - Global batch size: {global_bsz}")
    log(f"  - Micro batch/GPU: {stage.micro_bsz}")
    log(f"  - Grad accum: {stage.grad_accum}")
    log(f"  - Steps/epoch: {steps_per_epoch}")
    log(f"  - Epochs: {epochs}")
    log(f"  - Total steps: {total_steps}")
    log(f"  - LR: {stage.lr}")
    log(f"  - Warmup ratio: {stage.warmup_ratio}")

    # Torch 2.4 safeguard (avoid fsdp2 grad-clip import)
    torch_ge_24 = _torch_ge_24()

    ov = [
        f"data.train_files={str(train_pq)}",
        f"data.val_files={str(val_pq)}",
        f"data.train_batch_size={global_bsz}",
        f"data.micro_batch_size_per_gpu={stage.micro_bsz}",
        f"data.max_length={stage.max_len}",
        "data.truncation=left",
        "data.multiturn.enable=true",
        "data.multiturn.messages_key=messages",
        "+data.dataloader_num_workers=32",
        "+data.prefetch_factor=4",
        "+data.persistent_workers=true",
        "+data.pin_memory=true",
        f"data.val_max_samples={val_max}",
        f"model.partial_pretrain={base_for_stage}",
        "model.trust_remote_code=true",
        "model.fsdp_config.model_dtype=bfloat16",
        "model.enable_gradient_checkpointing=true",
        f"model.lora_rank={cfg.lora_rank}",
        f"model.lora_alpha={cfg.lora_alpha}",
        f"+model.lora_dropout={cfg.lora_dropout}",
        f"model.target_modules={lora_targets}",
        "use_remove_padding=true",
        "ulysses_sequence_parallel_size=1",
        f"optim.lr={stage.lr}",
        f"+optim.max_lr={stage.lr}",
        f"optim.lr_warmup_steps_ratio={stage.warmup_ratio}",
        "optim.lr_scheduler=cosine",
        "optim.min_lr_ratio=0.1",
        "optim.weight_decay=0.01",
        # clip_grad set below for torch>=2.4
        f"optim.total_training_steps={total_steps}",
        f"trainer.default_local_dir={str(stage_root)}",
        "trainer.project_name=sft_3stage_verl",
        f"trainer.experiment_name=stage{stage.num}",
        f"trainer.total_epochs={epochs}",
        f"trainer.total_training_steps={total_steps}",
        "trainer.logger=[console]",
        f"trainer.seed={cfg.seed}",
        f"trainer.save_freq={stage.save_freq}",
        "+trainer.log_freq=1",
        "+trainer.eval_freq=1",
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={ngpu}",
        # (no checkpoint override here)
    ]
    if train_max_samples_override is not None:
        ov.append(f"data.train_max_samples={train_max_samples_override}")

    if torch_ge_24:
        # Avoid VERL's fsdp2 grad-clip path on torch>=2.4
        ov.append("model.strategy=fsdp")  # <-- fixed (no '+')
        ov.append("optim.clip_grad=0.0")
        log("Torch>=2.4 detected: forcing model.strategy=fsdp and disabling grad clipping.")

    return ov

def _watchdog(p: subprocess.Popen, max_hours: float, grace_secs: int = 300):
    start = time.time()
    cap = int(max_hours * 3600)
    while True:
        if p.poll() is not None: 
            return
        if int(time.time() - start) >= cap:
            log("⏱ Time budget reached, sending graceful shutdown signal...")
            try: os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except: pass
            time.sleep(grace_secs // 2)
            if p.poll() is None:
                log("⏱ Force terminating...")
                try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except: pass
            return
        time.sleep(10)

def _run_stage_with_verl(ngpu: int, env: dict, overrides: List[str], max_hours: float) -> None:
    cmd = ["torchrun", f"--nproc_per_node={ngpu}", "-m", "verl.trainer.fsdp_sft_trainer"] + overrides
    log("=" * 80)
    log("Launching VERL FSDP SFT trainer:")
    log("  " + " ".join(cmd[:6]) + " ...")
    log("=" * 80)
    p = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
    t = threading.Thread(target=_watchdog, args=(p, max_hours), daemon=True)
    t.start()
    ret = p.wait()
    t.join(timeout=5)
    if ret != 0:
        raise SystemError(f"VERL trainer exited with code {ret}")
    log("✓ Training completed successfully")

def _find_latest_global_step_dir(stage_root: Path) -> Optional[Path]:
    if not stage_root.exists(): 
        return None
    cands = [p for p in stage_root.glob("global_step_*") if p.is_dir()]
    if not cands:
        cands = [p for p in stage_root.glob("checkpoint-*") if p.is_dir()] or \
                [p for p in stage_root.glob("step_*") if p.is_dir()]
    if not cands:
        return None
    def _step(d: Path):
        try: return int(d.name.split("_")[-1].replace("checkpoint-", "").replace("step-", ""))
        except: return -1
    cands.sort(key=_step)
    latest = cands[-1]
    log(f"✓ Found checkpoint: {latest.name}")
    return latest

def _merge_with_model_merger(local_dir: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "verl.model_merger", "merge",
           "--backend", "fsdp",
           "--local_dir", str(local_dir),
           "--target_dir", str(target_dir)]
    log("=" * 80)
    log("Merging FSDP checkpoint to HuggingFace format:")
    log(f"  Source: {local_dir.name}")
    log(f"  Target: {target_dir.name}")
    log("=" * 80)
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        raise SystemError(f"Model merger failed (exit {ret})")
    log("✓ Model merger completed successfully")

# -------------------- post-stage eval & generation --------------------
def _build_hf_eval_dataset(eval_jsonl: Path, max_len: int, base_model: str):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    log(f"Building evaluation dataset (max_len={max_len})")
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    ds = load_dataset("json", data_files=str(eval_jsonl), split="train")

    def _fmt(ex):
        msgs = ex.get("messages") or ex.get("conversations")
        if msgs is None: return {"text": None}
        md = ex.get("metadata")
        if md is not None and isinstance(msgs, list) and msgs and msgs[0].get("role") != "system":
            md_str = md if isinstance(md, str) else json.dumps(md, ensure_ascii=False)
            msgs = [{"role": "system", "content": f"[METADATA]\n{md_str}"}] + msgs
        try:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            txt = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs])
        return {"text": txt}

    ds = ds.map(_fmt)
    ds = ds.filter(lambda ex: ex["text"] is not None and len(ex["text"]) > 0)

    def _tok(batch):
        return tok(batch["text"], truncation=True, max_length=max_len, padding=False, return_attention_mask=True)
    ds = ds.map(_tok, batched=True, remove_columns=ds.column_names)
    return tok, ds

def _final_eval_and_gen(hf_model_dir: Path, eval_jsonl: Path, max_len: int,
                        n_eval: int, n_gen: int, max_new: int, out_stage_dir: Path):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoTokenizer
    from datasets import load_dataset

    log("=" * 80)
    log(f"Post-stage evaluation: {hf_model_dir.name}")
    log("=" * 80)
    
    tokenizer, eval_ds = _build_hf_eval_dataset(eval_jsonl, max_len, str(hf_model_dir))
    if n_eval > 0 and len(eval_ds) > n_eval:
        eval_ds = eval_ds.select(range(n_eval))

    log(f"Loading model from {hf_model_dir.name}")
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_model_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    dl = DataLoader(eval_ds, batch_size=4, shuffle=False, collate_fn=collator)
    
    model.eval()
    loss_sum = 0
    nseq = 0
    
    log(f"Evaluating on {len(eval_ds):,} samples...")
    with torch.no_grad():
        for i, batch in enumerate(dl):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss_sum += float(out.loss) * batch["input_ids"].shape[0]
            nseq += batch["input_ids"].shape[0]
            if (i + 1) % 100 == 0:
                log(f"  Processed {nseq:,} / {len(eval_ds):,} samples")
    
    avg_loss = loss_sum / max(1, nseq)
    ppl = math.exp(avg_loss) if avg_loss < 30 else float("inf")
    
    with open(out_stage_dir / "final_eval.json", "w") as f:
        json.dump({"final_eval_loss": avg_loss, "final_eval_ppl": ppl, "num_samples": nseq}, f, indent=2)
    log(f"✓ Final eval: loss={avg_loss:.6f}  ppl={ppl:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    log(f"Generating {n_gen} sample outputs...")
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_model_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.eval()
    raw = load_dataset("json", data_files=str(eval_jsonl), split="train")
    n = min(len(raw), max(1, n_gen))
    sub = raw.select(range(n))
    gens = []
    tok = AutoTokenizer.from_pretrained(str(hf_model_dir), use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    for idx, ex in enumerate(sub):
        msgs = ex.get("messages") or ex.get("conversations")
        if msgs is None: 
            continue
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs])
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=True, temperature=0.8, top_p=0.95,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        gens.append({"messages": msgs, "generation": text})
        if (idx + 1) % 20 == 0:
            log(f"  Generated {idx + 1} / {n}")
    
    with open(out_stage_dir / "generations.jsonl", "w", encoding="utf-8") as f:
        for g in gens: f.write(json.dumps(g, ensure_ascii=False) + "\n")
    log(f"✓ Wrote {len(gens)} generations to generations.jsonl")
    del model
    torch.cuda.empty_cache()
    gc.collect()

# -------------------- orchestrator --------------------
def _run_stage(cfg: RunCfg, ngpu: int, env: dict, stage: StageCfg,
               train_pq: Path, val_pq: Path, base_for_stage: str, stage_dir: Path) -> Path:
    log("\n" + "=" * 80)
    log(f"STAGE {stage.num} / 3: Context={stage.max_len} tokens, Hours={stage.hours}, "
        f"MaxSteps={cfg.max_steps if cfg.max_steps else 'none'}")
    log("=" * 80 + "\n")
    
    train_rows = _parquet_num_rows(train_pq)
    log(f"Training on {train_rows:,} rows from {train_pq.name}")
    if train_rows <= 0:
        raise SystemError(f"Train parquet '{train_pq}' has 0 rows; cannot proceed.")
    
    overrides = _build_overrides(cfg, ngpu, train_pq, val_pq, train_rows, stage, stage_dir, base_for_stage)
    _run_stage_with_verl(ngpu, env, overrides, stage.hours + 0.25)

    log("Searching for checkpoint...")
    latest = _find_latest_global_step_dir(stage_dir)
    if not latest:
        raise SystemError(f"No VERL checkpoint found in {stage_dir} (global_step_* missing).")

    merged_dir = stage_dir / f"merged_model_stage{stage.num}"
    _merge_with_model_merger(latest, merged_dir)
    if not (merged_dir / "config.json").exists():
        raise SystemError(f"Merged model missing config.json in {merged_dir}")
    log(f"✓ Merged model saved to: {merged_dir}")

    _final_eval_and_gen(merged_dir, cfg.eval_file, stage.max_len,
                        cfg.final_eval_samples, cfg.gen_samples, cfg.gen_max_new, stage_dir)
    log(f"\n✓ Stage {stage.num} complete!\n")
    return merged_dir

def main():
    log("=" * 80)
    log("3-STAGE SFT TRAINING WITH VERL + FlashAttention-2")
    log("=" * 80 + "\n")
    
    cfg = _parse_args()
    log("Checking dependencies...")
    for m, h in (
        ("torch", "pip install torch"),
        ("flash_attn", "pip install 'flash-attn>=2.6.0'"),
        ("verl", "pip install verl"),
        ("peft", "pip install peft"),
        ("transformers", "pip install transformers"),
        ("datasets", "pip install datasets"),
        ("pyarrow", "pip install pyarrow")
    ):
        _require(m, h)

    env, ngpu = _setup_env()
    if ngpu < 1: 
        raise SystemError("No CUDA devices visible.")
    
    log(f"\n{'=' * 80}")
    log(f"Configuration:")
    log(f"  Base model: {cfg.base_model}")
    log(f"  GPUs: {ngpu}")
    log(f"  Output: {cfg.output_dir}")
    log(f"  Stage 1: {cfg.len1} tokens, {cfg.stage1_hours}h")
    log(f"  Stage 2: {cfg.len2} tokens, {cfg.stage2_hours}h")
    log(f"  Stage 3: {cfg.len3} tokens, {cfg.stage3_hours}h")
    log(f"  Total time budget: {cfg.stage1_hours + cfg.stage2_hours + cfg.stage3_hours}h")
    log(f"  Max steps cap: {cfg.max_steps if cfg.max_steps else 'none'}")
    log(f"{'=' * 80}\n")
    
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = cfg.output_dir / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    train_pq = data_root / "train.parquet"
    val_pq = data_root / "val.parquet"
    
    # TRAIN parquet
    if not train_pq.exists(): 
        _convert_jsonl_to_parquet(cfg.train_file, train_pq)
    else:
        rows = _parquet_num_rows(train_pq)
        if rows <= 0:
            log(f"⚠ Detected existing but empty train parquet: {train_pq.name} — rebuilding from JSONL")
            train_pq.unlink(missing_ok=True)
            _convert_jsonl_to_parquet(cfg.train_file, train_pq)
        else:
            log(f"✓ Using existing train parquet: {train_pq.name}")
    if _parquet_num_rows(train_pq) <= 0:
        raise SystemError(f"Train parquet still has 0 rows after rebuild. "
                          f"Check JSONL at {cfg.train_file} has valid 'messages' arrays.")

    # VAL parquet
    if not val_pq.exists():   
        _convert_jsonl_to_parquet(cfg.eval_file, val_pq)
    else:
        log(f"✓ Using existing validation parquet: {val_pq.name}")

    # Buckets
    if cfg.use_buckets:
        stage_pq = _build_length_buckets(train_pq, cfg, data_root)
        train_s1, train_s2, train_s3 = stage_pq[1], stage_pq[2], stage_pq[3]
        if _parquet_num_rows(train_s1) == 0: log("⚠ Stage 1 bucket empty, using full dataset"); train_s1 = train_pq
        if _parquet_num_rows(train_s2) == 0: log("⚠ Stage 2 bucket empty, using full dataset"); train_s2 = train_pq
        if _parquet_num_rows(train_s3) == 0: log("⚠ Stage 3 bucket empty, using full dataset"); train_s3 = train_pq
    else:
        log("Using full dataset for all stages (bucketing disabled)")
        train_s1 = train_s2 = train_s3 = train_pq

    # Stage configs
    s1 = StageCfg(1, cfg.len1, cfg.stage1_hours, cfg.stage1_lr, cfg.stage1_warmup,
                  cfg.stage1_micro_bsz, cfg.stage1_grad_accum, cfg.save_freq1)
    s2 = StageCfg(2, cfg.len2, cfg.stage2_hours, cfg.stage2_lr, cfg.stage2_warmup,
                  cfg.stage2_micro_bsz, cfg.stage2_grad_accum, cfg.save_freq2)
    s3 = StageCfg(3, cfg.len3, cfg.stage3_hours, cfg.stage3_lr, cfg.stage3_warmup,
                  cfg.stage3_micro_bsz, cfg.stage3_grad_accum, cfg.save_freq3)

    t0 = time.time()

    stage1_dir = cfg.output_dir / "stage1"
    base1 = cfg.base_model
    merged1 = _run_stage(cfg, ngpu, env, s1, train_s1, val_pq, base1, stage1_dir)

    stage2_dir = cfg.output_dir / "stage2"
    base2 = str(merged1)
    merged2 = _run_stage(cfg, ngpu, env, s2, train_s2, val_pq, base2, stage2_dir)

    stage3_dir = cfg.output_dir / "stage3"
    base3 = str(merged2)
    merged3 = _run_stage(cfg, ngpu, env, s3, train_s3, val_pq, base3, stage3_dir)

    total_h = (time.time() - t0) / 3600.0
    summary = {
        "completed_at": _now(),
        "total_hours": total_h,
        "configuration": {
            "base_model": cfg.base_model,
            "gpus": ngpu,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "max_steps": cfg.max_steps,
        },
        "stages": {
            "stage1": {"context_length": cfg.len1, "hours": cfg.stage1_hours,
                       "learning_rate": cfg.stage1_lr,
                       "merged_model": str((stage1_dir / "merged_model_stage1").resolve()),
                       "final_eval": str((stage1_dir / "final_eval.json").resolve()),
                       "generations": str((stage1_dir / "generations.jsonl").resolve())},
            "stage2": {"context_length": cfg.len2, "hours": cfg.stage2_hours,
                       "learning_rate": cfg.stage2_lr,
                       "merged_model": str((stage2_dir / "merged_model_stage2").resolve()),
                       "final_eval": str((stage2_dir / "final_eval.json").resolve()),
                       "generations": str((stage2_dir / "generations.jsonl").resolve())},
            "stage3": {"context_length": cfg.len3, "hours": cfg.stage3_hours,
                       "learning_rate": cfg.stage3_lr,
                       "merged_model": str((stage3_dir / "merged_model_stage3").resolve()),
                       "final_eval": str((stage3_dir / "final_eval.json").resolve()),
                       "generations": str((stage3_dir / "generations.jsonl").resolve())},
        },
        "note": "Use any merged_model as an RL initial policy or for inference."
    }
    summary_path = cfg.output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    log("\n" + "=" * 80)
    log("✓ PIPELINE COMPLETE!")
    log("=" * 80)
    log(f"Total time: {total_h:.2f} hours")
    log(f"Summary: {summary_path}")
    log(f"\nFinal model: {merged3}")
    log("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        log(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
