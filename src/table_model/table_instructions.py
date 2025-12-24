#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Stage-2 SFT messages from merged JSONL where each line has:
  {
    "id": "...",
    "split": "...",
    "text": "...",
    "schema": {"table_id": "...", "columns": [{"name": ...}, ...], "n_cols": ...},
    "table":  {"table_id": "...", "columns": [{"name": ...}, ...], "data": [[...], ...]}
  }

Output (one JSONL per input):
  {
    "id": "...",
    "messages": [
      {"role": "system", "content": SYSTEM_PI2},
      {"role": "user",    "content": USER_PROMPT},
      {"role": "assistant","content": "<NDJSON>"}   # one JSON object per line
    ],
    "metadata": {"table_id": "...", "split": "..."}
  }

Usage:
  python stage2_make_messages_min.py \
    --inputs /path/merged_train.jsonl /path/merged_validation.jsonl /path/merged_test.jsonl \
    --out-dir /path/sft_stage2_messages

Notes:
- Also supports .jsonl.gz
- Fails fast on any inconsistency (mismatched table_id, missing columns, bad JSON)
"""

from __future__ import annotations
import argparse, json, io, gzip
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

# ---------- I/O ----------

def open_maybe_gzip(path: Path, mode: str = "rt", encoding: str = "utf-8"):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, mode.replace("t", "")), encoding=encoding)
    return path.open(mode=mode, encoding=encoding)

def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open_maybe_gzip(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise SystemExit(f"[fatal] {path}:{i}: invalid JSON: {e}")

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"[ok] wrote {n} rows -> {path}")
    return n

def infer_out_path(inp: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = inp.name
    if name.endswith(".jsonl.gz"):
        name = name[:-8]
    elif name.endswith(".jsonl"):
        name = name[:-6]
    return out_dir / f"{name}_stage2_messages.jsonl"

# ---------- Prompt blocks (tight + minimal) ----------

SYSTEM_PI2 = (
    "You convert documents into tabular data strictly under a provided JSON schema. "
    "Output ONLY JSON Lines (one JSON object per row), with EXACT columns and order as the schema, and no commentary."
)

POLICY = (
    "[POLICY]\n"
    "- Use only facts explicitly present in the document; no guessing.\n"
    "- Output exactly the columns in [SCHEMA] and in the same order.\n"
    "- If a value is missing, output an empty string \"\" for that cell.\n"
    "- Emit ONLY JSONL (one JSON object per line). No arrays, no markdown, no comments.\n"
    "- Trim leading/trailing whitespace; preserve internal spacing and case.\n"
)

OUTPUT_FORMAT = (
    "[OUTPUT_FORMAT]\n"
    "Emit ONLY JSONL, one JSON object per row, keys exactly as in [SCHEMA] and in the same order."
)

TASK = (
    "[TASK]\n"
    "Fill the table under [SCHEMA] using facts from [DOCUMENT]. For missing values, output \"\". "
    "Ensure each line is valid JSON."
)

def detect_structure(names: List[str]) -> str:
    return "kv_multi" if len(names) == 2 and names[0].lower() == "slot" and names[1].lower() == "value" else "flat_single_row"

def to_prompt_schema(raw_schema: Dict[str, Any]) -> Dict[str, Any]:
    cols = raw_schema.get("columns") or []
    names = [c.get("name", "") for c in cols]
    return {
        "schema_id": raw_schema.get("table_id", "auto"),
        "structure": detect_structure(names),
        "description": "Schema induced from the gold table.",
        "fields": [{"name": n, "type": "string", "required": True} for n in names],
    }

def build_user_message(text: str, p_schema: Dict[str, Any]) -> str:
    # <|document|> is included so your quick-sanity tool can parse the [SCHEMA] block.
    return "\n\n".join([
        POLICY,
        "[SCHEMA]\n"   + json.dumps(p_schema, ensure_ascii=False),
        "<|document|>",
        "[DOCUMENT]\n" + (text or ""),
        OUTPUT_FORMAT,
        TASK,
    ])

# ---------- NDJSON rendering ----------

def schema_cols(raw_schema: Dict[str, Any]) -> List[str]:
    return [c.get("name", "") for c in (raw_schema.get("columns") or [])]

def table_cols(table: Dict[str, Any]) -> List[str]:
    return [c.get("name", "") for c in (table.get("columns") or [])]

def render_ndjson(schema_col_order: List[str], table: Dict[str, Any]) -> str:
    tcols = table_cols(table)
    index = {name: i for i, name in enumerate(tcols)}
    lines: List[str] = []
    for row in (table.get("data") or []):
        obj: Dict[str, Any] = {}
        for k in schema_col_order:
            i = index.get(k, -1)
            val = "" if i < 0 or i >= len(row) or row[i] is None else str(row[i]).strip()
            obj[k] = val
        line = json.dumps(obj, ensure_ascii=False)
        # sanity: key order must match schema order
        if list(json.loads(line).keys()) != schema_col_order:
            raise SystemExit("[fatal] internal: NDJSON key order mismatch")
        lines.append(line)
    return "\n".join(lines)

# ---------- Validation ----------

def validate_example(ex: Dict[str, Any]) -> None:
    ex_id = ex.get("id")
    if not isinstance(ex_id, str) or not ex_id:
        raise SystemExit("[fatal] missing non-empty 'id'")
    if not isinstance(ex.get("text"), str) or not ex["text"].strip():
        raise SystemExit(f"[fatal] id={ex_id}: missing/empty 'text'")
    sch = ex.get("schema") or {}
    tbl = ex.get("table") or {}
    if sch.get("table_id") != tbl.get("table_id"):
        raise SystemExit(f"[fatal] id={ex_id}: schema.table_id != table.table_id")
    sc, tc = schema_cols(sch), table_cols(tbl)
    missing = [c for c in sc if c not in tc]
    if missing:
        raise SystemExit(f"[fatal] id={ex_id}: table missing schema columns: {missing}")

# ---------- Per-file processing ----------

def process_file(inp: Path, out_dir: Path) -> Path:
    out_path = infer_out_path(inp, out_dir)
    out_rows: List[Dict[str, Any]] = []

    for ex in read_jsonl(inp):
        validate_example(ex)
        sch, tbl = ex["schema"], ex["table"]
        p_schema = to_prompt_schema(sch)
        user = build_user_message(ex["text"], p_schema)
        ndjson = render_ndjson([f["name"] for f in p_schema["fields"]], tbl)

        out_rows.append({
            "id": ex["id"],
            "messages": [
                {"role": "system", "content": SYSTEM_PI2},
                {"role": "user",    "content": user},
                {"role": "assistant","content": ndjson},
            ],
            "metadata": {"table_id": sch.get("table_id"), "split": ex.get("split")}
        })

    write_jsonl(out_path, out_rows)
    return out_path

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser("Create Stage-2 SFT instruction messages from merged JSONL.")
    ap.add_argument("--inputs", nargs="+", type=Path, required=True,
                    help="One or more merged .jsonl (or .jsonl.gz) files.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory to place <input>_stage2_messages.jsonl outputs.")
    args = ap.parse_args()

    for p in args.inputs:
        if not p.exists():
            raise SystemExit(f"[fatal] input not found: {p}")
        outp = process_file(p, args.out_dir)
        print(f"[OK] {p.name} → {outp}")

if __name__ == "__main__":
    main()
