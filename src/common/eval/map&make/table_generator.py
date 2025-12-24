#!/usr/bin/env python3
"""
Infer filled tables from atomic statements + schema using Llama 3.1 8B Instruct with vLLM.

Expected input (JSONL, one object per line), e.g. (output of schema stage):
{
  "id": 0,
  "summary": "...",
  "table": {...},                      # optional original Rotowire table
  "atomic_statements": [...],          # from atomization stage
  "schema_raw_output": "<schema text>" # human-readable schema with row/column headers
}

Output (JSONL, one object per line):
{
  "id": 0,
  "summary": "...",
  "table": {...},                      # untouched original if present
  "atomic_statements": [...],
  "schema_raw_output": "<schema text>",
  "tables_raw_output": "<full table model text>",
  "tables": {
    "Team": {
      "headers": ["Team", "Wins", "Losses", "Total Points"],
      "rows": [
        ["Thunder", 16, 17, null],
        ["Suns", 18, null, 134]
      ]
    },
    "Player": {
      "headers": ["Player", "Points", "Minutes Played"],
      "rows": [
        ["Kevin Durant", 44, 40]
      ]
    }
  }
}
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# === PROMPT (EXACTLY as you provided) =======================================

TABLE_PROMPT = """You are an expert in converting unstructured text into structured tables. Your task is to process a series of atomic statements and update a set of pre-defined table schemas. Note that you can be given more than one table to update.
Follow the steps below to ensure accurate and complete table updates.

***TASK***:
**Given**:
*Statements*: A sequence of atomic statements.
*Schema*: A json object with table names and their row headers and column headers of the respective tables.

**Your goal is to**:
Process each statement one by one.
Identify the correct set of table, row and column headers and the cell at that index to update based on the statement.
Update or add values to the tables accordingly.

***OUTPUT FORMAT***:
<REASONING STEPS>

### Final Output Tables:
### <TABLE NAME>
| | <Column Header 1> | ... |
| <Row Header 1> | <Cell Value for (Row Header 1, Column Header 1)> | ... |
...



***REASONING STEPS***:
Follow the given algorithm thoroughly,

**ALGORITHM**
For each statement in the input:
    Identify Table:
        Determine the correct table to be updated based on the table.
    Identify Row and Column:
        Determine which set of row and columns headers have to be updated based on this table.
    Update the Table:
        If no value exists, update the value of the cell as per the statement.


**Sample Input**:
Statements:
The Oklahoma City Thunder's record is 16 wins.
The Oklahoma City Thunder's record is 17 losses.
The Phoenix Suns' record is 18 wins.
The Oklahoma City Thunder defeated the Phoenix Suns in overtime on Wednesday.
The Phoenix Suns scored 134 points.
Oklahoma City Thunder has won three of their last four games.
Kevin Durant returned to play after a six-game absence due to an ankle sprain.
Kevin Durant scored a season-high 44 points in the game.
Kevin Durant played 40 minutes during the game.

Schema:
{
    "Team": {
    "row_headers": ["Thunder", "Suns"],
    "column_headers": ["Wins", "Losses", "Total points"]
    }
    "Player": {
        "row_headers": ["Kevin Durant"],
        "col_headers": ["Points", "Minutes played"]
    }
}


**Step 1**:
Initial tables (Empty Tables):

### Team:
| Team | Wins | Losses | Total Points |
| Thunder | None | None | None |
| Suns | None | None | None |

### Player:
| Player | Points | Minutes Played |
| Kevin Durant | None | None |


**Step 2**:
*Statement processed*: 
The Oklahoma City Thunder's record is 16 wins.
*Updates*: 
    Table: Team
    Row: Thunder
    Column: Wins
    Value: 16

*Statements processed*:
The Oklahoma City Thunder's record is 17 losses.
*Update*:
    Table: Team
    Row: Thunder
    Column: Losses
    Value: 17

*Statements processed*:
The Phoenix Suns' record is 18 wins.
*Update*: 
    Table: Team
    Row: Suns 
    Column: Wins
    Value: 18

*Statements processed*:
The Oklahoma City Thunder defeated the Phoenix Suns in overtime on Wednesday.
*Update*:
    Table: Team
    Row: Thunder, Suns
    Column: Not found , Hence no update required.

*Statements processed*:
The Phoenix Suns scored 134 points.
*Update*:
    Table: Team
    Row: Suns 
    Column: Total Points 
    Value: 18

*Statements processed*:
Oklahoma City Thunder has won three of their last four games.
*Update*: 
    Table: Team
    Row: Thunder
    Column: Not found, hence no update required

*Statements processed*:
Kevin Durant returned to play after a six-game absence due to an ankle sprain.
*Update*: 
    Table: Player
    Row: Kevin Durant
    Column: Not found, hence no update required

*Statements processed*:
Kevin Durant returned to play after a six-game absence due to an ankle sprain.
*Update*: 
    Table: Player
    Row: Kevin Durant
    Column: Not found, hence no update required

*Statements processed*:
Kevin Durant scored a season-high 44 points in the game.
*Update*: 
    Table: Player
    Row: Kevin Durant
    Column: Points
    Value = 44

*Statements processed*:
Kevin Durant played 40 minutes during the game.
*Update*: 
    Table: Player
    Row: Kevin Durant
    Column: Minutes Played
    Value = 40


### Final Output Tables:

### Team
| Team | Wins | Losses | Total Points |
| Thunder | 16 | 17| None |
| Suns | 18 | None | 134 |

### Player
| Player | Points | Minutes Played |
| Kevin Durant | 44 | 40 |


***FINAL CHECKLIST***:
Follow these guidelines to generate tables and return the final state of the table after processing all the statements. 
Ensure all sentences are processed and for every statement return the update and revised state of the updated cells as shown in the example. Return the final table in the exact specified format starting with ### Final Table.
Do not generate the final table directly in any case. 
No need to generate the intermediate table states, just return the final table at the end.
Ensure the table is concise, well-structured, and contains all information from the input.

***Final Output Instructions***:
1. Handle Missing Data:
    If a column value is not present in the statements, keep it as None.
2. Structural Integrity:
    Do not add or remove any rows or columns unless explicitly instructed by the data.
    Ensure uniformity in the format of data across the table.
3. Table formatting:
    Use "|" to separate cells

Provide the *OUTPUT* with *REASONING STEPS* in the specified format only.
"""

# Extra control message (does NOT change the prompt)
TABLE_CONTROL_MSG = (
    "You MUST strictly follow the OUTPUT FORMAT. "
    "Keep the REASONING STEPS section concise (no more than 10 short bullet points) "
    "and ALWAYS include the '### Final Output Tables:' section with properly "
    "formatted markdown tables for every table in the schema. "
    "If you do not output '### Final Output Tables:' your answer is incorrect."
)


def setup_hf_cache(hf_home: Optional[str]):
    """
    Configure Hugging Face cache directories under a base path.
    This affects both transformers and vLLM (via HF_HOME/HF_HUB_CACHE).
    """
    if hf_home is None:
        return

    hf_home = os.path.abspath(os.path.expanduser(hf_home))
    os.makedirs(hf_home, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "transformers")
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")

    print(f"[setup_hf_cache] Using HF cache at: {hf_home}", flush=True)


def load_engine_and_tokenizer(
    model_name: str,
    tensor_parallel_size: int,
    hf_home: Optional[str],
    gpu_memory_utilization: float,
):
    """
    Load vLLM engine (LLM) and a HuggingFace tokenizer for chat templating.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but vLLM currently expects GPU. "
            "Please run on a GPU node."
        )

    download_dir = hf_home if hf_home is not None else None

    print(
        f"[load_engine_and_tokenizer] Loading tokenizer for {model_name} "
        f"(download_dir={download_dir})",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=True,
    )

    print(
        f"[load_engine_and_tokenizer] Initializing vLLM LLM for {model_name} "
        f"with tensor_parallel_size={tensor_parallel_size}, "
        f"gpu_memory_utilization={gpu_memory_utilization}",
        flush=True,
    )
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        download_dir=download_dir,
        trust_remote_code=False,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=True,
    )

    print("[load_engine_and_tokenizer] LLM engine ready.", flush=True)
    return llm, tokenizer


def build_table_prompt(
    tokenizer,
    atomic_statements: List[str],
    schema_text: str,
) -> str:
    """
    Build a Llama-3.1-style chat prompt for table generation
    using atomic statements + schema text as input.
    """
    statements_text = "\n".join(
        f"{i+1}. {stmt}" for i, stmt in enumerate(atomic_statements)
    )

    schema_block = schema_text.strip() if isinstance(schema_text, str) else ""

    user_content = (
        "You are given the following atomic statements and schema.\n\n"
        "### Statements:\n"
        f"{statements_text}\n\n"
        "### Schema:\n"
        f"{schema_block}\n\n"
        "Using ONLY these statements and this schema, update the tables as instructed."
    )

    messages = [
        {"role": "system", "content": TABLE_PROMPT},
        {"role": "system", "content": TABLE_CONTROL_MSG},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def _convert_cell(cell: str):
    """
    Convert a single table cell from string to Python type:
    - "None" / "null" / "" -> None
    - integers -> int
    - otherwise -> original string
    """
    s = cell.strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        return s


def parse_tables_markdown(raw_output: str) -> Optional[Dict[str, Any]]:
    """
    Parse markdown tables under '### Final Output Tables:' into a structured dict.
    """
    lines = raw_output.splitlines()
    n = len(lines)

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("### Final Output Tables"):
            start_idx = i + 1
            break

    if start_idx is None:
        return None

    tables: Dict[str, Dict[str, Any]] = {}
    current_table_name: Optional[str] = None
    current_headers: Optional[List[str]] = None
    current_rows: List[List[Any]] = []

    i = start_idx
    while i < n:
        line = lines[i].strip()

        if line.startswith("### "):
            if current_table_name and current_headers:
                tables[current_table_name] = {
                    "headers": current_headers,
                    "rows": current_rows,
                }
            current_table_name = line.lstrip("#").strip()
            current_headers = None
            current_rows = []
            i += 1
            continue

        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            cells = [c for c in parts if c != ""]
            if not cells:
                i += 1
                continue

            if current_headers is None:
                current_headers = cells
            else:
                converted = [_convert_cell(c) for c in cells]
                current_rows.append(converted)

        i += 1

    if current_table_name and current_headers:
        tables[current_table_name] = {
            "headers": current_headers,
            "rows": current_rows,
        }

    if not tables:
        return None

    return {"tables": tables}


def generate_table_batch(
    llm: LLM,
    tokenizer,
    records: List[Dict[str, Any]],
    sampling_params: SamplingParams,
    atomic_field: str,
    schema_field: str,
) -> List[str]:
    """
    Run table generation for a batch of records with vLLM.
    Each record must have:
      - list of atomic statements under `atomic_field`
      - schema text under `schema_field`
    """
    prompts: List[str] = []
    for r in records:
        atomic_statements = r.get(atomic_field) or []
        if not isinstance(atomic_statements, list):
            raise ValueError(
                f"Record id={r.get('id')} has non-list '{atomic_field}': {type(atomic_statements)}"
            )

        schema_text = r.get(schema_field)
        if not isinstance(schema_text, str):
            raise ValueError(
                f"Record id={r.get('id')} missing string '{schema_field}' for schema text."
            )

        prompts.append(build_table_prompt(tokenizer, atomic_statements, schema_text))

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    completions: List[str] = []
    for i, out in enumerate(outputs):
        if not out.outputs:
            raise RuntimeError(f"No outputs returned for record index {i}")
        completions.append(out.outputs[0].text.strip())

    return completions


def infer_tables_file(
    input_path: Path,
    output_path: Path,
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
    batch_size: int,
    log_interval: int,
    atomic_field: str,
    schema_field: str,
):
    """
    Read JSONL with atomic statements + schema text from input_path,
    run table inference in batches, and write JSONL with table info to output_path.
    """
    print(
        f"[infer_tables_file] Starting table inference\n"
        f"  input:         {input_path}\n"
        f"  output:        {output_path}\n"
        f"  batch_size:    {batch_size}\n"
        f"  log_interval:  {log_interval}\n"
        f"  atomic_field:  {atomic_field}\n"
        f"  schema_field:  {schema_field}",
        flush=True,
    )

    total_processed = 0
    batch: List[Dict[str, Any]] = []

    with input_path.open() as f_in, output_path.open("w") as f_out:
        for line_idx, line in enumerate(f_in):
            raw_line = line.strip()
            if not raw_line:
                continue

            obj = json.loads(raw_line)

            if atomic_field not in obj:
                raise ValueError(
                    f"Line {line_idx}: expected '{atomic_field}' field with atomic statements."
                )
            if schema_field not in obj:
                raise ValueError(
                    f"Line {line_idx}: expected '{schema_field}' field with schema text."
                )

            record = obj
            record["_line_idx"] = line_idx
            batch.append(record)

            if len(batch) >= batch_size:
                print(
                    f"[infer_tables_file] Generating tables for batch "
                    f"{total_processed}–{total_processed + len(batch) - 1} "
                    f"(size={len(batch)})...",
                    flush=True,
                )
                completions = generate_table_batch(
                    llm=llm,
                    tokenizer=tokenizer,
                    records=batch,
                    sampling_params=sampling_params,
                    atomic_field=atomic_field,
                    schema_field=schema_field,
                )

                for rec, completion in zip(batch, completions):
                    tables_obj = parse_tables_markdown(completion)
                    out_obj = dict(rec)
                    out_obj.pop("_line_idx", None)
                    out_obj["tables_raw_output"] = completion
                    out_obj["tables"] = tables_obj.get("tables") if tables_obj else None
                    f_out.write(json.dumps(out_obj) + "\n")

                f_out.flush()
                total_processed += len(batch)

                print(
                    f"[infer_tables_file] Completed batch. "
                    f"Total processed so far: {total_processed}",
                    flush=True,
                )

                if total_processed % log_interval == 0:
                    print(
                        f"[infer_tables_file] PROGRESS: {total_processed} records processed.",
                        flush=True,
                    )

                batch = []

        if batch:
            print(
                f"[infer_tables_file] Generating tables for final batch "
                f"{total_processed}–{total_processed + len(batch) - 1} "
                f"(size={len(batch)})...",
                flush=True,
            )
            completions = generate_table_batch(
                llm=llm,
                tokenizer=tokenizer,
                records=batch,
                sampling_params=sampling_params,
                atomic_field=atomic_field,
                schema_field=schema_field,
            )

            for rec, completion in zip(batch, completions):
                tables_obj = parse_tables_markdown(completion)
                out_obj = dict(rec)
                out_obj.pop("_line_idx", None)
                out_obj["tables_raw_output"] = completion
                out_obj["tables"] = tables_obj.get("tables") if tables_obj else None
                f_out.write(json.dumps(out_obj) + "\n")

            f_out.flush()
            total_processed += len(batch)
            print(
                f"[infer_tables_file] Final batch done. Total processed: {total_processed}",
                flush=True,
            )

    print("[infer_tables_file] Table inference complete ✅", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Path to input JSONL file with at least 'atomic_statements' and "
            "'schema_raw_output' fields per line."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file with inferred tables.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--hf_home",
        type=str,
        default=None,
        help="Base directory for Hugging Face caches (e.g., /scratch/$USER/hf-cache or ~/hf-cache).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p for nucleus sampling (ignored if temperature=0.0 but harmless).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to shard the model across (vLLM tensor_parallel_size).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help=(
            "Fraction of each GPU memory vLLM may use (0.0–1.0). "
            "If you see 'Free memory on device ... is less than desired', "
            "lower this value."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of records to process per vLLM batch (higher = faster, more VRAM).",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Log a progress message every N records.",
    )
    parser.add_argument(
        "--atomic_field",
        type=str,
        default="atomic_statements",
        help="Field containing the list of atomic statements in the input JSONL.",
    )
    parser.add_argument(
        "--schema_field",
        type=str,
        default="schema_raw_output",
        help="Field containing the schema text (the human-readable schema output).",
    )
    args = parser.parse_args()

    setup_hf_cache(args.hf_home)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[main] Using model: {args.model_name}\n"
        f"[main] Input: {input_path}\n"
        f"[main] Output: {output_path}\n"
        f"[main] Tensor parallel size: {args.tensor_parallel_size}\n"
        f"[main] GPU mem utilization: {args.gpu_memory_utilization}\n"
        f"[main] Atomic field: {args.atomic_field}\n"
        f"[main] Schema field: {args.schema_field}",
        flush=True,
    )

    llm, tokenizer = load_engine_and_tokenizer(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        hf_home=args.hf_home,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    infer_tables_file(
        input_path=input_path,
        output_path=output_path,
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        atomic_field=args.atomic_field,
        schema_field=args.schema_field,
    )


if __name__ == "__main__":
    main()
