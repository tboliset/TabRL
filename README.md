# TABRL: TabRL-Summarizer

Two-stage text-to-table generation with LLMs, plus a MONA-style (multi-policy) RL blueprint and a full **T3-style atomic-statement pipeline** (atomize → schema → table).

This repository contains the code, training scripts, and docs for **TabRL-Summarizer**, a system that:

- Reads unstructured text (sports recaps, biographies, etc.).
- **Stage 1 (π₁)**: predicts an explicit JSON **schema**.
- **Stage 2 (π₂)**: uses the schema to generate a **table** as NDJSON.
- Provides a complete pipeline of helper models that convert between **text, atomic statements, schemas, and tables**.
- Designs a **multi‑policy RL extension** where schema and table policies get their own rewards.

The layout below matches the committed Python files in this repo.

---

## Repository structure

```text
TABRL/
├─ docs/                            # Project reports and design docs
│  ├─ Stratos_Final_Report.pdf
│  ├─ Work_Till_Now_NLP.pdf
│  └─ RL-pipeline.pdf
├─ examples/
│  └─ sample_input/
│     └─ sample_data.jsonl          # Tiny demo dataset (text + schema/table)
├─ src/
│  ├─ common/
│  │  └─ eval/
│  │     ├─ map&make/               # Baseline map-and-make evaluation code
│  │     ├─ tab_eval/               # Table evaluation utilities (metrics, parsing)
│  │     ├─ entailment_eval.py      # (alt location) wrapper for NLI-based checks
│  │     └─ tabunroll_infer.py      # (alt location) helpers for T3-style pipelines
│  ├─ RL/                           # MONA RL design & (optional) prototype code
│  ├─ schema_model/
│  │  ├─ schema_instructions.py     # Prompt / message templates for π₁
│  │  └─ schema_train_sft.py        # Small wrapper around train_sft_stage_one.py
│  └─ table_model/
│     ├─ table_instructions.py      # Prompt / message templates for π₂
│     └─ table_train_sft.py         # Small wrapper around train_sft_stage_two.py
├─ train_sft_stage_one.py           # Stage‑1 SFT trainer (text → schema)
├─ train_sft_stage_two.py           # Stage‑2 SFT trainer (text + schema → table)
├─ build_instructions_stage_one.py  # Build Stage‑1 SFT chat messages
├─ build_instructions_stage_two.py  # Build Stage‑2 SFT chat messages
├─ atomize_llama31.py               # (atomize_llama31) Text → atomic statements
├─ statement-gen.py                 # Atomic statements → schema text
├─ table-gen.py                     # Atomic statements + schema → completed table
├─ tabunroll_infer.py               # Table → atomic statements (T3 “unroll”)
├─ entailment_eval.py               # NLI-based entailment / contradiction scorer
└─ README.md
```

> Depending on where you place `entailment_eval.py` and `tabunroll_infer.py`, they may live either in `src/common/eval` or the repo root. The README assumes they exist and are importable from your Python path; adjust paths if you move files around.

---

## Data format

Most scripts in this repo expect **JSONL** (one JSON object per line). In general an example looks like:

```json
{
  "id": 0,
  "summary": "Free-form text describing the game, person, or entity...",
  "table": {...},                     // optional ground-truth table
  "schema": {...},                    // optional structured schema
  "atomic_statements": [...],         // optional list of atomic statements
  "...": "..."
}
```

Individual scripts specialize this schema (see script‑specific sections below).

The file `examples/sample_input/sample_data.jsonl` contains a tiny, self-contained demo that you can run end‑to‑end without any external datasets.

---

## Stage‑1 SFT: text → schema (π₁)

### Purpose

Train a schema policy π₁ that maps documents to compact JSON schemas using Llama‑3.1 + LoRA with VERL/FSDP on H100s.

### Script: `train_sft_stage_one.py`

- 3‑stage schedule (short → medium → long context).
- Uses VERL’s trainer with FSDP + FlashAttention‑2.
- Builds Parquet caches from JSONL.
- Supports explicit time budgets (`--stage*_hours`) and a **hard cap** on steps via `--max_steps`.
- Designed for 2× H100 but can be run with fewer GPUs with smaller batch sizes.

**Key arguments (subset):**

- `--train_file` : path to Stage‑1 SFT training JSONL.
- `--eval_file`  : path to validation JSONL.
- `--output_dir` : directory to write checkpoints / logs.
- `--base_model` : HF model id (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`).
- `--max_steps`  : hard upper bound on total training steps.
- LoRA options: `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--lora_targets`.

**Typical usage:**

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python train_sft_stage_one.py \
  --train_file /path/to/schema_sft_train.jsonl \
  --eval_file  /path/to/schema_sft_valid.jsonl \
  --output_dir /path/to/checkpoints/schema_sft \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max_steps 200
```

### Script: `build_instructions_stage_one.py`

Builds **Stage‑1 SFT chat messages** from raw merged JSONL.

- Input JSONL: lines with at least `id`, document text, and a gold table/metadata.
- Output JSONL: one object per line with:
  - `"id"`
  - `"messages"`: `[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", ...}, {"role": "assistant", ...}]`
  - `"metadata"`: light metadata for debugging.

It contains a large, domain‑agnostic **policy** for schema induction (naming rules, evidence requirements, constraints, missing‑value handling, etc.), which is rendered into the `[POLICY]` section of the user prompt.

**Typical usage (conceptual):**

```bash
python build_instructions_stage_one.py \
  --input  /path/to/merged_train.jsonl \
  --output /path/to/schema_stage1_messages_train.jsonl
```

(Use the actual CLI flags from the script when you run it.)

---

## Stage‑2 SFT: text + schema → table (π₂)

### Purpose

Train a table policy π₂ that maps (document, schema) pairs to completed tables emitted as NDJSON.

### Script: `train_sft_stage_two.py`

- Mirrors Stage‑1 trainer:
  - 3‑stage schedule.
  - VERL FSDP + FlashAttention‑2.
  - Parquet caching and length bucketing.
- Adds **post‑stage sanity checks**: LM loss, sample generations, JSONL validation (schema keys / order / types).

**Key arguments (subset):**

- `--train_file`, `--eval_file`, `--output_dir`, `--base_model`, `--max_steps` (same semantics as Stage‑1).
- Additional options for bucketing, context length, etc., as documented in the script.

**Typical usage:**

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python train_sft_stage_two.py \
  --train_file /path/to/table_sft_train.jsonl \
  --eval_file  /path/to/table_sft_valid.jsonl \
  --output_dir /path/to/checkpoints/table_sft \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max_steps 200
```

### Script: `build_instructions_stage_two.py`

Builds **Stage‑2 SFT chat messages** from merged JSONL that already contains text, schema, and table.

- **Input (per line):**

  ```json
  {
    "id": "...",
    "split": "...",
    "text": "...",
    "schema": {"table_id": "...", "columns": [...], "n_cols": ...},
    "table":  {"table_id": "...", "columns": [...], "data": [[...], ...]}
  }
  ```

- **Output (per line):**

  ```json
  {
    "id": "...",
    "messages": [
      {"role": "system", "content": SYSTEM_PI2},
      {"role": "user",    "content": USER_PROMPT},
      {"role": "assistant","content": "<NDJSON>"}
    ],
    "metadata": {"table_id": "...", "split": "..."}
  }
  ```

**Typical usage (from docstring):**

```bash
python build_instructions_stage_two.py \
  --inputs /path/merged_train.jsonl /path/merged_validation.jsonl \
  --out-dir /path/sft_stage2_messages
```

Supports `.jsonl` and `.jsonl.gz`, and fails fast on invalid JSON.

---

## Atomic statement / T3‑style pipeline

These scripts implement a T3‑style pipeline that moves between **text, atomic statements, schemas, and tables** using Llama‑3.1 8B + vLLM.

### 1. Text → atomic statements: `atomize_llama31.py`

- Uses vLLM’s `LLM` engine for fast batched inference.
- Builds chat prompts with the Llama‑3.1 chat template.
- Reads JSONL like:

  ```json
  {"id": 0, "summary": "...", "table": {...}}
  ```

- Writes JSONL like:

  ```json
  {
    "id": 0,
    "summary": "...",
    "table": {...},
    "raw_output": "<full model text>",
    "atomic_statements": [
      "The Atlanta Hawks' record is 46 wins.",
      "..."
    ]
  }
  ```

Important flags (see script):

- `--input` / `--output`
- `--base-model` (HF id for Llama‑3.1‑8B‑Instruct)
- vLLM settings: `--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-num-seqs`, etc.

### 2. Atomic statements → schema text: `statement-gen.py`

Given atomic statements, infer a table‑like schema using Llama‑3.1 + vLLM.

- **Input JSONL** (from `atomize_llama31.py`):

  ```json
  {
    "id": 0,
    "summary": "...",
    "table": {...},
    "raw_output": "<atomization model text>",
    "atomic_statements": ["...", "..."]
  }
  ```

- **Output JSONL:**

  ```json
  {
    "id": 0,
    "summary": "...",
    "table": {...},
    "atomic_statements": [...],
    "schema_raw_output": "<full schema model text>"
  }
  ```

This `schema_raw_output` is a human‑readable text schema (row/column headers) that can be parsed or post‑processed into a JSON schema for π₂.

### 3. Atomic statements + schema → completed table: `table-gen.py`

Fill in tables from atomic statements plus schema using Llama‑3.1 + vLLM.

- **Input JSONL:** (typically after `statement-gen.py` + a schema parser)

  ```json
  {
    "id": 0,
    "summary": "...",
    "table": {...},                      // optional original table
    "atomic_statements": [...],
    "schema_raw_output": "<schema text>"
  }
  ```

- **Output JSONL:** one object per line with the same fields plus a generated table representation (see the script for the exact key names).

The script contains a `TABUNROLL_PROMPT` that describes how to interpret schemas and atomic statements when writing out table rows.

### 4. Table → atomic statements: `tabunroll_infer.py`

The “reverse” direction: given a table, ask Llama‑3.1 to write atomic statements describing every fact in the table.

- Uses the same `TABUNROLL_PROMPT` family as `table-gen.py`.
- Helpful for:
  - Converting gold tables to statements for T3 evaluation.
  - Debugging the schema/table models by seeing what the model “thinks” the table says.

**Input / output:** JSONL with table fields; see the script for the precise shape.

---

## Entailment and evaluation helpers

### Script: `entailment_eval.py`

- Loads **RoBERTa‑large‑MNLI** from Hugging Face.
- Computes textual entailment / contradiction between pairs of strings (e.g., atomic statements vs. source text, or statement vs. table verbalization).
- Can be used as an additional evaluator or reward component.

Key behaviours:

- Automatically chooses `cuda` if available, else `cpu`.
- Reads JSONL inputs, scores each pair, and writes out entailment probabilities / labels.

### `src/common/eval`

The `src/common/eval` folder contains shared evaluation utilities:

- `map&make/` – Baseline “map‑and‑make” table generation pipeline.
- `tab_eval/` – Parsing / metric code for comparing predicted vs. gold tables.
- `tabunroll_infer.py` & `entailment_eval.py` can be imported from here as part of larger eval scripts.

---

## Running an end‑to‑end demo

A small, conceptual example of how pieces connect (you can adapt paths to your exact setup):

```bash
# 1. Atomize summaries into statements
python atomize_llama31.py \
  --input  examples/sample_input/sample_data.jsonl \
  --output outputs/atomized.jsonl \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct

# 2. From atomic statements, infer a schema text
python statement-gen.py \
  --input  outputs/atomized.jsonl \
  --output outputs/schema_raw.jsonl \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct

# 3. From atomic statements + schema text, generate tables
python table-gen.py \
  --input  outputs/schema_raw.jsonl \
  --output outputs/generated_tables.jsonl \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct
```

Once you have `generated_tables.jsonl`, you can evaluate it against gold tables using the utilities in `src/common/eval/tab_eval` or by writing a small wrapper script.

---

## RL (MONA blueprint)

The `src/RL` folder contains the **design and prototype code** for multi‑policy RL:

- Schema policy (π₁) and table policy (π₂) are treated as **separate agents**.
- Each has its own reward:
  - **Schema rewards**: JSON validity, unique / non‑empty columns, allowed types, coverage vs. gold.
  - **Table rewards**: JSON validity, schema adherence, type checks, table metrics (cell‑level F1, overlaps).
- Rewards are **non‑aggregating**: you optimize π₁ on schema rewards and π₂ on table rewards, instead of mixing them.
- The design is compatible with VERL‑style group relative preference training with KL regularization.

Full design notes are in `docs/RL-pipeline.pdf` and `Stratos_Final_Report.pdf`.

---

## Project documents (for reviewers)

If you’re a professor or recruiter skimming the repo, here is the recommended reading order:

1. **README.md** (this file) – high‑level idea, code structure, and quickstart.
2. **docs/Stratos_Final_Report.pdf** – paper‑style full report (method, experiments, RL design).
3. **docs/Work_Till_Now_NLP.pdf** – accessible narrative explanation of the project.
4. **docs/RL-pipeline.pdf** – deep dive into the RL training pipeline and engineering details.

---

## Citation

If you use this work, you can cite it as:

```bibtex
@misc{tabrl-summarizer,
  title        = {TabRL-Summarizer: Two-Stage Text-to-Table Generation with MONA Multi-Policy Reinforcement Learning},
  author       = {Patil, Arya and Bolisetty, Thanishka and Sharma, Simran and Pradhan, Hriday and Mehboob, Mohammed Usman},
  howpublished = {\url{https://github.com/tboliset/TABRL}},
  year         = {2025}
}
```

(Replace the GitHub URL with your actual repository link.)

---

## Contact

Maintainer: **Mohammed Usman Mehboob**  
Email: `musmanme@asu.edu`

Feel free to reach out or open an issue if you have questions about the code, training scripts, or RL design.
