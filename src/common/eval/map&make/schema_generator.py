#!/usr/bin/env python3
"""
Infer table-like schemas from atomic statements using Llama 3.1 8B Instruct with vLLM.

Expected input (JSONL, one object per line), e.g. (output of atomize_llama31.py):
{
  "id": 0,
  "summary": "...",
  "table": {...},
  "raw_output": "<atomization model text>",
  "atomic_statements": [
    "The Oklahoma City Thunder's record is 16 wins.",
    "The Oklahoma City Thunder's record is 17 losses.",
    ...
  ]
}

Output (JSONL, one object per line):
{
  "id": 0,
  "summary": "...",
  "table": {...},
  "atomic_statements": [...],
  "schema_raw_output": "<full schema model text>",
  "schema": {
    "schema_id": "team_player_stats",
    "structure": "object",
    "fields": [
      {"name": "team_name", "type": "string", "required": true},
      {"name": "wins", "type": "integer", "required": false},
      ...
    ]
  }
}
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# === PROMPT (EXACTLY as you provided) =======================================

SCHEMA_PROMPT = """You are an expert at defining structural tables by identifying the relevant column headers and row headers from text.

***TASK***:
Given a set of atomic text statements, extract row and column headers to create a table schema.

***INSTRUCTIONS***:
Read the statements carefully to identify all attributes, entities, and data points mentioned, whether explicitly stated or implicitly implied.
Determine the row headers (primary keys) and column headers required to represent the data comprehensively and concisely:
Row headers are the unique identifiers for individual rows (key entities).
Column headers are the attributes of the primary keys that represent different aspects or data points.
Include all explicit and implicit data points, ensuring no relevant information is overlooked. 
Pay close attention to numerical data, even if it is presented within comparative statements or descriptions of events or related to specific categories or time periods mentioned in the text. 
Explicit numerical data must always be captured as attributes where appropriate. 
Implicit data points or recurring attributes must also be included.
Avoid adding actions as column headers but extract any data points associated with them.
Ensure that all numerical values are captured as attributes, even if they are related to specific time periods or events within the context. When encountering comparative statements or ratios like "X of Y", ensure you capture both 'X' and 'Y' as potentially distinct and relevant data points if they represent different aspects of an attribute.
Be attentive to granular details and avoid focusing solely on general or aggregate values if more specific data points are available in the text.

***OUTPUT FORMAT***:
<REASONING STEPS>

### Final Schema:
{
    "<Table name>": {
        "row_headers": ["Row Header 1", ...],
        "column_headers": ["Column Header 1", ...]
    }
    ...
}


***REASONING STEPS***:
**Sample input**:
The Oklahoma City Thunder's record is 16 wins.
The Oklahoma City Thunder's record is 17 losses.
The Phoenix Suns' record is 18 wins.
The Phoenix Suns' record is 16 losses.
The Oklahoma City Thunder defeated the Phoenix Suns in overtime on Wednesday.
The Oklahoma City Thunder scored 137 points.
The Phoenix Suns scored 134 points.
Oklahoma City Thunder has won three of their last four games.
Kevin Durant returned to play after a six-game absence due to an ankle sprain.
Kevin Durant scored a season-high 44 points in the game.
Kevin Durant played 40 minutes during the game.

**Step1 - Identify the context from all the statements to generate a short description**
Thought: This is a summary of a basketball game played between the Oklahoma City Thunder and the Phoenix Suns and gives all the relevant statistics about the players and the games. Every statement is either about the team or one of the players hence it would be ideal to create to separate tables for them. One table for the teams, and one for the players.

**Step2 - Create a empty list of row and column headers for the tables. This list would be updated as we keep on processing the statements and will keep adding relevant column and row headers to the list.**
*Intermediate output*: 
{
    "Team": {
    "row_headers": [],
    "column_headers": [], 
    }
    "Player": {
        "row_headers": [],
        "col_headers": []
    }
}

**Step 3 - Process statements one by one and add relevant headers if not already present in the list.**

*Statements processed*:
1. The Oklahoma City Thunder's record is 16 wins.
*Schema update*:
    - Update in "Team" table
        - Row added: "Thunder"
        - Column added: "Wins"

2. The Oklahoma City Thunder's record is 17 losses.
*Schema update*:
    - Update in "Team" table
        - Row added: None ("Thunder" is already present in the schema)
        - Column added: "Losses"

3. The Phoenix Suns' record is 18 wins.
*Schema update*:
    - Update in "Team" table
        - Row added: "Suns"
        - Column added: None ("Wins" is already present in the schema)

4. The Phoenix Suns' record is 16 losses.
*Schema update*:
    - Update in "Team" table
        - Row added: None ("Suns" is already present in the schema)
        - Column added: None ("Losses" is already present in the schema)

5. The Oklahoma City Thunder defeated the Phoenix Suns in overtime on Wednesday.
*Schema update*:
    - Update in "Team" table
        - Row added: None
        - Column added: None

6. The Oklahoma City Thunder scored 137 points.
*Schema update*:
    - Update in "Team" table
        - Row added: None ("Thunder" is already present in the schema)
        - Column added: "Total points"

7. The Phoenix Suns scored 134 points.
*Schema update*:
    - Update in "Team" table
        - Row added: None ("Suns" is already present in the schema)
        - Column added: None ("Total points" is already present in the schema)


8. Oklahoma City Thunder has won three of their last four games.
*Schema update*:
    - Update in "Team" table
        - Row added: None
        - Column added: None


9. Kevin Durant returned to play after a six-game absence due to an ankle sprain.
*Schema update*:
    - Update in "Team" table
        - Row added: None
        - Column added: None 
    - Update in "Player" table
        - Row added: "Kevin Durant"
        - Column added: None


10. Kevin Durant scored a season-high 44 points in the game.
*Schema update*:
    - Update in "Team" table
        - Row added: None
        - Column added: None 
    - Update in "Player" table
        - Row added: None ("Kevin Durant" is already present in the schema)
        - Column added: "Points"

10. Kevin Durant played 40 minutes during the game.
*Schema update*:
    - Update in "Team" table
        - Row added: None
        - Column added: None 
    - Update in "Player" table
        - Row added: None ("Kevin Durant" is already present in the schema)
        - Column added: "Minutes played"


### Final Schema:
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

***Final Output Instructions***:
1. As shown in the illustration above, for *every given statement* return the updates done to the schema and generate the Team Table and Player Table schema.
2. Do not return schema directly in any case.
3. Provide the *OUTPUT* with *REASONING STEPS* in the specified format only.
"""


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


def build_schema_prompt(tokenizer, atomic_statements: List[str]) -> str:
    """
    Build a Llama-3.1-style chat prompt for schema generation
    using the atomic statements as input.
    """
    statements_text = "\n".join(
        f"{i+1}. {stmt}" for i, stmt in enumerate(atomic_statements)
    )

    user_content = (
        "You are given the following atomic statements extracted from a text.\n\n"
        "### Atomic Statements:\n"
        f"{statements_text}\n\n"
        "Using these atomic statements ONLY, derive the table-level schema as instructed."
    )

    messages = [
        {"role": "system", "content": SCHEMA_PROMPT},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def _snake_case(name: str) -> str:
    # lower, replace non-alphanum with underscores, squash repeats, strip edges
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.lower())
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _build_schema_from_final_schema_dict(final_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn a dict like:
      {"Team": {"row_headers": [...], "column_headers": [...]}, ...}
    into:
      {"schema": { "schema_id": "...", "structure": "object", "fields": [...] }}
    """
    table_names = list(final_schema.keys())
    if not table_names:
        return {"schema": None}

    if len(table_names) == 1:
        schema_id = f"{table_names[0].lower()}_schema"
    else:
        schema_id = "_".join(t.lower() for t in table_names) + "_schema"

    fields_by_name: Dict[str, Dict[str, Any]] = {}

    # Add a name field per table (e.g. team_name, player_name)
    for t in table_names:
        fname = f"{t.lower()}_name"
        if fname not in fields_by_name:
            fields_by_name[fname] = {
                "name": fname,
                "type": "string",
                "required": True if t == table_names[0] else False,
            }

    # Add fields from column_headers / col_headers
    for t, entry in final_schema.items():
        if not isinstance(entry, dict):
            continue
        col_headers = entry.get("column_headers") or entry.get("col_headers") or []
        for header in col_headers:
            if not isinstance(header, str):
                continue
            fname = _snake_case(header)
            if not fname:
                continue
            if fname in fields_by_name:
                continue

            # Very simple heuristic: ID/name-ish → string, otherwise integer.
            lower = fname.lower()
            if any(k in lower for k in ["name", "id", "team", "player", "city"]):
                ftype = "string"
            else:
                ftype = "integer"
            fields_by_name[fname] = {
                "name": fname,
                "type": ftype,
                "required": False,
            }

    schema = {
        "schema_id": schema_id,
        "structure": "object",
        "fields": list(fields_by_name.values()),
    }
    return {"schema": schema}


def parse_schema_json(raw_output: str) -> Optional[Dict[str, Any]]:
    """
    Extract a machine-readable schema object from the model output.

    Priority:
    1) If the model ever outputs a line starting with '{"schema"', parse that as JSON.
    2) Otherwise, parse the "### Final Schema:" block and synthesize {"schema": {...}}.
    """
    lines = raw_output.splitlines()

    # --- 1) Look for explicit {"schema": ...} JSON (future-proof) -------------
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('{"schema"'):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                # Try to accumulate multiple lines until braces balance
                json_text = stripped
                open_braces = stripped.count("{") - stripped.count("}")
                j = idx + 1
                while open_braces > 0 and j < len(lines):
                    extra = lines[j]
                    json_text += "\n" + extra
                    open_braces += extra.count("{") - extra.count("}")
                    j += 1
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass  # fall through to Final Schema parsing

    # --- 2) Parse the ### Final Schema: block --------------------------------
    marker = "### Final Schema:"
    if marker not in raw_output:
        return None

    start_idx = raw_output.index(marker) + len(marker)
    # Find first '{' after marker
    brace_start = raw_output.find("{", start_idx)
    if brace_start == -1:
        return None

    # Collect until braces balance
    depth = 0
    end_pos = None
    for i in range(brace_start, len(raw_output)):
        ch = raw_output[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_pos = i
                break
    if end_pos is None:
        return None

    block = raw_output[brace_start : end_pos + 1]

    # Fix common non-JSON bits, e.g. missing comma between tables:
    #    }
    #    "Player": { ... }
    block_fixed = re.sub(r'}\s*\n\s*"', '},\n    "', block)

    try:
        final_schema = json.loads(block_fixed)
    except json.JSONDecodeError:
        # If we can't parse it, bail.
        return None

    if not isinstance(final_schema, dict):
        return None

    # Build our own {"schema": {...}} object from this dict
    return _build_schema_from_final_schema_dict(final_schema)


def generate_schema_batch(
    llm: LLM,
    tokenizer,
    records: List[Dict[str, Any]],
    sampling_params: SamplingParams,
    atomic_field: str,
) -> List[str]:
    """
    Run schema generation for a batch of records with vLLM.
    Each record must have a list of atomic statements under `atomic_field`.
    """
    prompts = []
    for r in records:
        atomic_statements = r.get(atomic_field) or []
        if not isinstance(atomic_statements, list):
            raise ValueError(
                f"Record id={r.get('id')} has non-list '{atomic_field}': {type(atomic_statements)}"
            )
        prompts.append(build_schema_prompt(tokenizer, atomic_statements))

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    completions: List[str] = []
    for i, out in enumerate(outputs):
        if not out.outputs:
            raise RuntimeError(f"No outputs returned for record index {i}")
        completions.append(out.outputs[0].text.strip())

    return completions


def infer_schema_file(
    input_path: Path,
    output_path: Path,
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
    batch_size: int,
    log_interval: int,
    atomic_field: str,
):
    """
    Read JSONL with atomic statements from input_path, run schema inference in batches,
    and write JSONL with schema information to output_path.
    """
    print(
        f"[infer_schema_file] Starting schema inference\n"
        f"  input:  {input_path}\n"
        f"  output: {output_path}\n"
        f"  batch_size: {batch_size}\n"
        f"  log_interval: {log_interval}\n"
        f"  atomic_field: {atomic_field}",
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

            record = obj
            record["_line_idx"] = line_idx
            batch.append(record)

            if len(batch) >= batch_size:
                print(
                    f"[infer_schema_file] Generating schemas for batch "
                    f"{total_processed}–{total_processed + len(batch) - 1} "
                    f"(size={len(batch)})...",
                    flush=True,
                )
                completions = generate_schema_batch(
                    llm=llm,
                    tokenizer=tokenizer,
                    records=batch,
                    sampling_params=sampling_params,
                    atomic_field=atomic_field,
                )

                for rec, completion in zip(batch, completions):
                    schema_obj = parse_schema_json(completion)
                    out_obj = dict(rec)  # shallow copy
                    out_obj.pop("_line_idx", None)
                    out_obj["schema_raw_output"] = completion
                    out_obj["schema"] = schema_obj.get("schema") if schema_obj else None
                    f_out.write(json.dumps(out_obj) + "\n")

                f_out.flush()
                total_processed += len(batch)

                print(
                    f"[infer_schema_file] Completed batch. "
                    f"Total processed so far: {total_processed}",
                    flush=True,
                )

                if total_processed % log_interval == 0:
                    print(
                        f"[infer_schema_file] PROGRESS: {total_processed} records processed.",
                        flush=True,
                    )

                batch = []

        # Final partial batch
        if batch:
            print(
                f"[infer_schema_file] Generating schemas for final batch "
                f"{total_processed}–{total_processed + len(batch) - 1} "
                f"(size={len(batch)})...",
                flush=True,
            )
            completions = generate_schema_batch(
                llm=llm,
                tokenizer=tokenizer,
                records=batch,
                sampling_params=sampling_params,
                atomic_field=atomic_field,
            )

            for rec, completion in zip(batch, completions):
                schema_obj = parse_schema_json(completion)
                out_obj = dict(rec)
                out_obj.pop("_line_idx", None)
                out_obj["schema_raw_output"] = completion
                out_obj["schema"] = schema_obj.get("schema") if schema_obj else None
                f_out.write(json.dumps(out_obj) + "\n")

            f_out.flush()
            total_processed += len(batch)
            print(
                f"[infer_schema_file] Final batch done. Total processed: {total_processed}",
                flush=True,
            )

    print("[infer_schema_file] Schema inference complete ✅", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file with 'atomic_statements' field per line (output of atomization).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file with inferred schema results.",
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
        help="Base directory for Hugging Face caches (e.g., /scratch/$USER/hf_home or ~/hf-cache).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
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
        help="Name of the field containing the list of atomic statements in the input JSONL.",
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
        f"[main] Atomic field: {args.atomic_field}",
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

    infer_schema_file(
        input_path=input_path,
        output_path=output_path,
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        atomic_field=args.atomic_field,
    )


if __name__ == "__main__":
    main()
