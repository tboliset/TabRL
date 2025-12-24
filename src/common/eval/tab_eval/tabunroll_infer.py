#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# === TABUNROLL PROMPT =========================================================
TABUNROLL_PROMPT = r"""You are a helpful AI assistant to help infer useful information from table structures. 
Task: You are given a table. Your goal is to write all the details conveyed in the table in the form of natural language statements. A statement is an atomic unit of information from the table.
Following the below instructions to do so:
1. Identify the column headers in the table.
2. Identify the various rows in the table.
3. From each row, identify meaningful and atomic pieces
of information that cannot be broken down further.
4. First, identify columns as primary key(s). A primary
key is the column or columns that contain values that
uniquely identify each row in a table.
5. If there is only one primary key identified, use it and add
information from each of the other columns one-by-one
to form meaningful statements.
6. If there are more than one primary key identified,
use them and add information from each of the other
columns one-by-one to form meaningful statements.
7. If no primary key is detected, then form the statements
by picking two columns at a time that make the most
sense in a meaningful manner.
8. In each of the above three cases, add information from
other columns (beyond the primary key column(s) or the
identified two columns in the absence of a primary key)
only if it is necessary to differentiate repeating entities.
9. Write all such statements in natural language.
10. Do not exclude any detail that is present in the given
table.
11. Give the supporting rows for each atomic statement.

Input table format:
| | <Column Header 1> | ... |
| <Row Header 1> | <Cell Value for (Row Header 1, Column Header 1)> | ... |
...

Final output format:
Table:
| | <Column Header 1> | ... |
| <Row Header 1> | <Cell Value for (Row Header 1, Column Header 1)> | ... |
...

Statements:
1. ...
2. ...
...

Rows:
1. | <Row Header 1> | <Cell Value for (Row Header 1, Column Header 1)> | ... |
...

Following are a few examples.
EXAMPLE 1
Table:
|Year|Competition|Venue|Position|Event|Notes|
|1966|European Indoor Games|Dortmund, West Germany|1st|400 m|47.9|
|1967|European Indoor Games|Prague, Czechoslovakia|2nd|400 m|48.6|

Statements:
1. European Indoor Games in 1966 occurred in Dortmund, West Germany.
2. 1st position was obtained in the 1966 European Indoor Games.
3. The 1966 European Indoor Games had a 400 m event.
4. 47.9 in the 1966 European Indoor Games.
5. European Indoor Games in 1967 occurred in Prague, Czechoslovakia.
6. 2nd position was obtained in the 1967 European Indoor Games.
7. The 1967 European Indoor Games had a 400 m event.
8. 48.6 in the 1967 European Indoor Games.

Rows:
1. |1966|European Indoor Games|Dortmund, West Germany|1st|400m|47.9|
2. |1967|European Indoor Games|Prague, Czechoslovakia|2nd|400m|48.6|

Example Bad Statements:
1. Koch came in 1st position in European Indoor Games in 1966 which occurred in Dortmund, West Germany.
2. 47.9 in European Indoor Games in 1966 which occurred in Dortmund, West Germany.
3. 2nd position in European Indoor Games in 1967 which occurred in Prague, Czechoslovakia.

EXAMPLE 2
Table:
|Year|Title|Role|Notes|
|2015|Kidnapped: The Hannah Anderson Story|Becca McKinnon|None|
|2015|Jem and the Holograms|Young Jerrica Benton|None|
|2015|Asomatous|Sophie Gibbs|None|
|2017|Unforgettable|Lily|None|
|2019|Our Friend|Molly|None|

Statements:
1. Kidnapped: The Hannah Anderson Story was filmed in 2015.
2. Isabella Rice played the role of Becca McKinnon in Kidnapped: The Hannah Anderson Story.
3. Jem and the Holograms was filmed in 2015.
4. Isabella Rice played the role of Young Jerrica Benton in the Jem and the Holograms.
5. Asomatous was filmed in 2015.
6. Isabella Rice played the role of Sophie Gibbs in Asomatous.
7. Unforgettable was filmed in 2017.
8. Isabella Rice played the role of Lily in Unforgettable.
9. Our Friend was filmed in 2019.
10. Isabella Rice played the role of Molly in Our Friend.

Rows:
1. |2015|Kidnapped: The Hannah Anderson Story|Becca McKinnon|None|
2. |2015|Jem and the Holograms|Young Jerrica Benton|None|
3. |2015|Asomatous|Sophie Gibbs|None|
4. |2017|Unforgettable|Lily|None|
5. |2019|Our Friend|Molly|None|

Example Bad Statements:
1. Isabella Rice played the role of Becca McKinnon in Kidnapped: The Hannah Anderson Story in 2015.
2. Jem and the Holograms was filmed in 2015 where Isabella Rice played the role of Young Jerrica Benton.
3. Isabella Rice played the role of Sophie Gibbs in Asomatous in 2015.
"""

CONTROL_MSG = (
    "You MUST strictly follow the Final output format. "
    "Do NOT add any extra sections. "
    "Always output 'Table:', 'Statements:' and 'Rows:' exactly as specified."
)


def setup_hf_cache(hf_home: Optional[str]):
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for vLLM.")
    download_dir = hf_home if hf_home is not None else None

    print(f"[load] Loading tokenizer for {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=False,
    )

    print(f"[load] Initializing vLLM for {model_name}", flush=True)
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
    print("[load] LLM ready.", flush=True)
    return llm, tokenizer


def table_to_markdown(headers: List[str], rows: List[List[Any]]) -> str:
    def fmt_cell(v):
        if v is None:
            return "None"
        return str(v)

    header_line = "|" + "|".join(headers) + "|"
    row_lines = []
    for row in rows:
        row_lines.append("|" + "|".join(fmt_cell(c) for c in row) + "|")
    return header_line + "\n" + "\n".join(row_lines)


def build_tabunroll_prompt(tokenizer, table_markdown: str) -> str:
    user_content = "Table:\n" + table_markdown
    messages = [
        {"role": "system", "content": TABUNROLL_PROMPT},
        {"role": "system", "content": CONTROL_MSG},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------- Pivot-style (single table object with columns/data) ---------------
def run_tabunroll_for_table_key(
    llm: LLM,
    tokenizer,
    records: List[Dict[str, Any]],
    table_key: str,  # "table_pred" or "table"
    out_dir: Path,
    sampling_params: SamplingParams,
):
    """
    Expects rec[table_key] to look like:
      {
        "columns": [{"name": ...}, ...],
        "data": [[...], ...]
      }
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts: List[Optional[str]] = []
    meta: List[Any] = []

    for rec in records:
        rid = rec["id"]
        tinfo = rec.get(table_key)

        if not isinstance(tinfo, dict):
            prompts.append(None)
            meta.append((rid, None))
            continue

        columns = tinfo.get("columns")
        data = tinfo.get("data")

        if not columns or not data:
            prompts.append(None)
            meta.append((rid, None))
            continue

        headers = [c.get("name", "") for c in columns]
        rows = data
        table_md = table_to_markdown(headers, rows)
        prompt = build_tabunroll_prompt(tokenizer, table_md)
        prompts.append(prompt)
        meta.append((rid, table_md))

    real_indices = [i for i, p in enumerate(prompts) if p is not None]
    real_prompts = [prompts[i] for i in real_indices]

    if not real_prompts:
        print(f"[run_tabunroll_for_table_key] No valid '{table_key}' tables found.")
        return

    outputs = llm.generate(real_prompts, sampling_params=sampling_params)

    for idx_in_batch, out in enumerate(outputs):
        full_text = out.outputs[0].text.strip()
        global_idx = real_indices[idx_in_batch]
        rec_id, _ = meta[global_idx]
        out_path = out_dir / f"{rec_id}.txt"
        with out_path.open("w") as f:
            f.write(full_text)


# ---------- Multi-table dict style (e.g., 'tables' baseline) ------------------
def run_tabunroll_for_multi_tables(
    llm: LLM,
    tokenizer,
    records: List[Dict[str, Any]],
    table_key: str,          # usually "tables" or a similar multi-table field
    out_dir: Path,
    sampling_params: SamplingParams,
):
    """
    Expects rec[table_key] to look like:
      {
        "<TableName1>": {"headers": [...], "rows": [[...], ...]},
        "<TableName2>": {"headers": [...], "rows": [[...], ...]},
        ...
      }

    Writes for each distinct table name T:
      out_dir/<t_lower>/<id>.txt
    """
    # Collect all table names that look valid across the dataset
    table_names = set()
    for rec in records:
        tinfo = rec.get(table_key)
        if not isinstance(tinfo, dict):
            continue
        for name, tbl in tinfo.items():
            if isinstance(tbl, dict) and tbl.get("headers") and tbl.get("rows"):
                table_names.add(name)

    if not table_names:
        print(f"[run_tabunroll_for_multi_tables] No valid tables found under key '{table_key}'.")
        return

    table_names = sorted(table_names)
    print(f"[run_tabunroll_for_multi_tables] Found table names: {table_names}")

    for tbl_name in table_names:
        safe_name = tbl_name.lower().replace(" ", "_")
        tbl_dir = out_dir / safe_name
        tbl_dir.mkdir(parents=True, exist_ok=True)

        prompts: List[Optional[str]] = []
        meta: List[Any] = []

        for rec in records:
            rid = rec["id"]
            tinfo = rec.get(table_key)

            if not isinstance(tinfo, dict):
                prompts.append(None)
                meta.append((rid, None))
                continue

            tbl = tinfo.get(tbl_name)
            if not isinstance(tbl, dict):
                prompts.append(None)
                meta.append((rid, None))
                continue

            headers = tbl.get("headers")
            rows = tbl.get("rows")

            if not headers or not rows:
                prompts.append(None)
                meta.append((rid, None))
                continue

            table_md = table_to_markdown(headers, rows)
            prompt = build_tabunroll_prompt(tokenizer, table_md)
            prompts.append(prompt)
            meta.append((rid, table_md))

        real_indices = [i for i, p in enumerate(prompts) if p is not None]
        real_prompts = [prompts[i] for i in real_indices]

        if not real_prompts:
            print(
                f"[run_tabunroll_for_multi_tables] No usable rows for table '{tbl_name}' "
                f"under key '{table_key}'."
            )
            continue

        print(
            f"[run_tabunroll_for_multi_tables] Tabunrolling table '{tbl_name}' "
            f"into {tbl_dir} ..."
        )
        outputs = llm.generate(real_prompts, sampling_params=sampling_params)

        for idx_in_batch, out in enumerate(outputs):
            full_text = out.outputs[0].text.strip()
            global_idx = real_indices[idx_in_batch]
            rec_id, _ = meta[global_idx]
            out_path = tbl_dir / f"{rec_id}.txt"
            with out_path.open("w") as f:
                f.write(full_text)


# ---------- Format detection ---------------------------------------------------
def detect_table_format(records: List[Dict[str, Any]], table_key: str) -> str:
    """
    Return one of:
      - "pivot"  => rec[table_key] has 'columns' & 'data'
      - "multi"  => rec[table_key] is a dict of tables with 'headers'/'rows'
      - "unknown"
    """
    for rec in records:
        tinfo = rec.get(table_key)
        if not isinstance(tinfo, dict):
            continue

        # Pivot-style: {columns: [...], data: [...]}
        if "columns" in tinfo and "data" in tinfo:
            return "pivot"

        # Multi-table dict style: {name: {headers: [...], rows: [...]}, ...}
        if any(
            isinstance(v, dict) and "headers" in v and "rows" in v
            for v in tinfo.values()
        ):
            return "multi"

    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="JSONL with 'id' and a table field ('table', 'table_pred', or 'tables') per example.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Output directory. "
            "For pivot tables: writes <id>.txt here. "
            "For multi-table dicts: writes <table_name>/<id>.txt."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--hf_home", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument(
        "--source_key",
        type=str,
        choices=["table_pred", "table", "tables"],
        default="table_pred",
        help=(
            "Key in each JSON record that contains the table(s). "
            "'table_pred'/'table' are usually single pivot tables with 'columns'/'data'. "
            "'tables' is a dict of named tables each with 'headers'/'rows'."
        ),
    )
    args = parser.parse_args()

    setup_hf_cache(args.hf_home)

    input_path = Path(args.input_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("[main] No records found in input JSONL.")
        return

    table_format = detect_table_format(records, args.source_key)
    print(f"[main] Detected table format for '{args.source_key}': {table_format}")

    if table_format == "unknown":
        raise ValueError(
            f"Could not detect table format for key '{args.source_key}'. "
            "Expected either pivot-style ('columns'/'data') or multi-table "
            "dict style with 'headers'/'rows' in child tables."
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

    if table_format == "pivot":
        print(
            f"[main] Tabunrolling PIVOT tables from '{args.source_key}' into {out_dir} ..."
        )
        run_tabunroll_for_table_key(
            llm,
            tokenizer,
            records,
            table_key=args.source_key,
            out_dir=out_dir,
            sampling_params=sampling_params,
        )
    else:  # "multi"
        print(
            f"[main] Tabunrolling MULTI tables from '{args.source_key}' into "
            f"{out_dir}/<table_name>/..."
        )
        run_tabunroll_for_multi_tables(
            llm,
            tokenizer,
            records,
            table_key=args.source_key,
            out_dir=out_dir,
            sampling_params=sampling_params,
        )

    print("[main] Done.")


if __name__ == "__main__":
    main()
