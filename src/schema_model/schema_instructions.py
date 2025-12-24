from __future__ import annotations
import argparse, hashlib, json, random, re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# -------------------------------
# (a) SYSTEM
# -------------------------------
SYSTEM_PROMPT = (
    "You induce minimal JSON schemas from documents. Output strictly valid JSON with no commentary."
)

# -------------------------------
# (b) POLICY — Full universal, domain-agnostic
# Rendered as a [POLICY] section by build_policy_block()
# -------------------------------
POLICY_LINES: List[str] = [
    # Naming
    "Use lower_snake_case (ASCII only). Trim; collapse whitespace/punctuation to '_' ; strip leading/trailing '_'.",
    "Resolve collisions by appending _2, _3, … in first-appearance order.",
    "Non-Latin keys: romanize when obvious; otherwise use field, field_2, … (do not translate values).",

    # Evidence only
    "Emit fields only when they are explicitly present in the document (table headers, KV keys, unambiguous inline labels). No synonyms, no background knowledge, no guessing.",
    "If two headers/keys normalize to the same name, treat as the same attribute; use the collision suffix only for truly distinct attributes that normalize identically.",

    # Field spec (minimal)
    "Every field must include exactly {name, type, required} where type ∈ {string, integer, number, boolean, date}.",
    "Add format only when type = \"date\".",
    "Add a top-level constraints object only when unambiguous (see Constraints).",

    # Typing (conservative, locale-robust)
    "Upgrade type only if all non-missing values fit it; otherwise use string.",
    "boolean: tokens ⊆ {true,false,yes,no,1,0,t,f} (case-insensitive).",
    "integer: ^[+-]?\\d+$ (trimmed).",
    "number: all tokens numeric (int/float), unit-free, with thousands/decimal separators used consistently (see METADATA locale hints).",
    "date: only ISO-like YYYY, YYYY-MM, or YYYY-MM-DD.",

    # Dates (simple constraint)
    "If type = \"date\", include format: \"YYYY|YYYY-MM|YYYY-MM-DD\" and constraints.patterns[<field>] = \"^\\d{4}(-\\d{2}(-\\d{2})?)?$\".",

    # Requiredness
    "required: true only if the field is present and non-empty for every applicable record in the chosen structure; otherwise false.",

    # Constraints (only when obvious; keep small)
    "constraints.enums.<field>: include when there are 2–10 distinct non-missing tokens after trim + case-fold. Never emit enum of size 1.",
    "constraints.patterns.<field>: include only for clearly pattern-constrained strings (e.g., the date regex). Do not guess complex regexes.",
    "Do not infer numeric ranges from observed min/max unless the document states them explicitly (then include min/max for that field).",

    # Missing values
    "Treat as missing (trimmed, case-insensitive): \"\", -, —, –, N/A, NA, None, Null, Unknown, TBD, and the na_token from METADATA. Exclude missings from typing/enum decisions.",

    # Structure (choose exactly one)
    "Select the single structure from METADATA.structure_candidates that best matches the document’s shape: "
    "{flat_single_row, row_grouped, kv_single, kv_multi, matrix_dense, multi_header, headerless_rows, ragged_rows, hierarchical_grouped}.",
    "If evidence is mixed, choose the structure capturing the majority of facts with least loss; tie-break to flat_single_row for single-entity prose, else row_grouped.",

    # Flattening (arrays/objects only when explicit)
    "Prefer flat fields.",
    "Arrays: use only when multiple values per attribute are explicit (repeated KV keys for the same attribute, consistent list markers/separators). Otherwise keep string.",
    "Objects: use only when ≥2 fields share a stable, explicit prefix repeated across records (e.g., address_street, address_city → object address with children street, city). Limit nesting to one level. When unsure, stay flat.",
    "Multi-headers/matrices: compose header layers with '_' to form field names (e.g., q1_revenue_usd), or select matrix_dense if that better preserves structure.",

    # Units & codes
    "If a unit/code appears in the header/key (e.g., price (USD)), encode it in the field name (price_usd). Do not invent conversions; preserve leading zeros in IDs/codes → keep as string.",

    # Ordering & determinism
    "Preserve encounter order: column headers left→right, then KV keys in reading order. When merging multiple sources within the same document, list columnar fields first. Output must be deterministic for identical input.",

    # Output
    "Return only a valid JSON object: schema_id, structure, optional static description, fields, and constraints (when present). No comments, markdown, or examples.",
]

def build_policy_block() -> str:
    """Render the [POLICY] section using the full universal policy (or a provided override)."""
    lines = [ln.strip() for ln in POLICY_LINES if ln and ln.strip()]
    return "[POLICY]\n" + "\n".join(f"- {ln}" for ln in lines)

STRUCTURE_CANDIDATES: List[str] = [
    "kv_single",
    "kv_multi",
    "flat_single_row",
    "row_grouped",
    "matrix_dense",
    "multi_header",
    "headerless_rows",
    "ragged_rows",
    "hierarchical_grouped",
]

def build_metadata(
    document_text: str,
    language: str = "auto",
    na_token: str = "",
    source_modality: str = "plain_text"
) -> Dict[str, Any]:
    doc_len = len(document_text or "")
    base = {
        "language": language,
        "script": "auto",
        "direction": "auto",
        "source_modality": source_modality,
        "na_token": na_token,
        "document_char_len": doc_len,
        "table_count": "auto",
        "structure_candidates": STRUCTURE_CANDIDATES,
        "table_hints": {
            "header_rows_max": 3,
            "header_cols_max": 2,
            "row_header_possible": True,
            "col_header_possible": True,
            "merged_cells_possible": True,
            "ragged_rows_possible": True,
            "multiple_tables_possible": True,
        },
        "locale": {
            "numeric": "auto",
            "decimal_separators": [".", ","],
            "thousand_separators": [",", ".", " "],
            "negative_patterns": ["-x", "(x)"],
            "percent_symbol": "%",
            "permille_symbol": "‰",
        },
        "parsing_hints": {
            "kv_markers": [":", "—", "–", "=", "→"],
            "list_markers": ["- ", "* ", "• ", "•\t", "▪ "],
            "section_markers": ["###", "##", "--", "__", "**", ":"],
        },
        "missing_tokens": ["", "-", "—", "–", "N/A", "NA", "None", "Null", "Unknown", "TBD"],
    }
    return base

def dropout_metadata(
    md: Dict[str, Any],
    p: float,
    keep_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    if p <= 0.0:
        return md
    keep = set(keep_keys or ["language", "source_modality"])
    out: Dict[str, Any] = {}
    for k, v in md.items():
        if (k in keep) or (random.random() >= p):
            out[k] = v
    return out

def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

NA_STRINGS = {"", "n/a", "na", "none", "null", "-", "—", "–", "unknown", "tbd"}

def is_missing(x: Any, na_token: str = "") -> bool:
    if x is None:
        return True
    s = str(x).strip()
    if na_token and s.lower() == na_token.lower():
        return True
    return s.lower() in NA_STRINGS

def normalize_cell(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()

def snake_case(name: str) -> str:
    if not name:
        return "field"
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip())
    s = s.lower().strip("_")
    return s or "field"

INT_RE   = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?$")
DATE_RE  = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
BOOL_TRUE  = {"true", "t", "yes", "y", "1"}
BOOL_FALSE = {"false", "f", "no", "n", "0"}

def infer_type(values: List[str], na_token: str = "") -> Tuple[str, Optional[str]]:
    vals = [v for v in values if not is_missing(v, na_token)]
    if not vals:
        return "string", None
    lows = [v.lower() for v in vals]
    if all(v in BOOL_TRUE.union(BOOL_FALSE) for v in lows):
        return "boolean", None
    if all(INT_RE.match(v) for v in vals):
        return "integer", None
    if all(DATE_RE.match(v) for v in vals):
        return "date", "YYYY|YYYY-MM|YYYY-MM-DD"
    if all(FLOAT_RE.match(v) or INT_RE.match(v) for v in vals):
        if all(re.fullmatch(r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?", v) or INT_RE.match(v) for v in vals):
            return "number", None
    return "string", None

def small_enum(values: List[str], na_token: str = "") -> Optional[List[str]]:
    vals = [v.strip() for v in values if not is_missing(v, na_token)]
    seen = set()
    uniq = []
    for v in vals:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(v)
    return uniq if 2 <= len(uniq) <= 10 else None

# -------------------------------
# Structure detection (no row data available in golden schemas)
# -------------------------------
KV_KEY_NAMES = {"slot", "key", "field", "attribute", "name", "label", "column", "property"}
KV_VAL_NAMES = {"value", "val", "text", "content", "data"}

def detect_structure(columns: List[Dict[str, Any]]) -> str:
    names = [snake_case(c.get("name", "")) for c in (columns or [])]
    if len(names) == 2 and names[0] in KV_KEY_NAMES and names[1] in KV_VAL_NAMES:
        return "kv_single"
    return "flat_single_row"

# -------------------------------
# Build assistant JSON from golden schema headers only (no row data)
# -------------------------------
def build_assistant_schema(
    example_id: str,
    columns: List[Dict[str, Any]],
    na_token: str = "",
) -> Dict[str, Any]:

    raw_names = [c.get("name", "") for c in (columns or [])]
    norm_names = [snake_case(n) for n in raw_names]

    # dedupe names with suffixes
    seen: Dict[str, int] = {}
    field_names: List[str] = []
    for n in norm_names:
        if n not in seen:
            seen[n] = 1
            field_names.append(n)
        else:
            seen[n] += 1
            field_names.append(f"{n}_{seen[n]}")

    # Without row values, be conservative
    fields: List[Dict[str, Any]] = [{"name": n, "type": "string", "required": False} for n in field_names]

    structure = detect_structure(columns)

    base = f"{example_id}|{structure}|{'|'.join(field_names)}"
    sid = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]

    schema: Dict[str, Any] = {
        "schema_id": f"auto.{sid}",
        "structure": structure,
        "description": "Minimal schema induced from the document.",
        "fields": fields,
    }
    return schema

def build_user_content(
    document_text: str,
    metadata_dropout: float = 0.0,
    language: str = "en",
    na_token: str = "",
    source_modality: str = "plain_text",
) -> str:

    policy_block = build_policy_block()

    md = build_metadata(
        document_text,
        language=language,
        na_token=na_token,
        source_modality=source_modality
    )
    md = dropout_metadata(md, metadata_dropout, keep_keys=["language", "source_modality"])
    metadata_block = "[METADATA]\n" + json.dumps(md, ensure_ascii=False)

    document_block = "[DOCUMENT]\n" + (document_text or "")
    task_block = "[TASK]\n" + SYSTEM_PROMPT

    msg = "\n\n".join([policy_block, metadata_block, document_block, task_block])

    msg = (
        "<|policy|>\n"    + policy_block   + "\n\n" +
        "<|metadata|>\n"  + metadata_block + "\n\n" +
        "<|document|>\n"  + document_block + "\n\n" +
        "<|task|>\n"      + task_block
    )
    return msg

def convert_file(
    inp: Path,
    out: Path,
    metadata_dropout: float = 0.0,
    language: str = "en",
    na_token: str = "",
    source_modality: str = "plain_text",
) -> None:

    def rows() -> Iterator[Dict[str, Any]]:
        for ex in read_jsonl(inp):
            text = (ex.get("text") or "").strip()
            schema = ex.get("schema") or {}
            if not text or not schema:
                continue

            user_content = build_user_content(
                text,
                metadata_dropout=metadata_dropout,
                language=language,
                na_token=na_token,
                source_modality=source_modality,
            )

            all_cols: List[Dict[str, Any]] = schema.get("columns") or []
            if not all_cols:
                continue

            assistant_json = build_assistant_schema(
                ex.get("id", ""),
                all_cols,
                na_token=na_token,
            )
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)},
                ]
            }
    write_jsonl(out, rows())

def infer_out_path(inp: Path, out_dir: Path) -> Path:
    name = inp.name.lower()
    if "train" in name:
        split = "train"
    elif "validation" in name or "dev" in name:
        split = "validation"
    elif "test" in name:
        split = "test"
    else:
        split = "data"
    return out_dir / f"schema_sft_{split}.jsonl"

def main():
    ap = argparse.ArgumentParser(description="Build instruction-tuning data (messages) for minimal schema induction.")
    ap.add_argument("--inputs", nargs="+", type=Path, required=True, help="Paths to *_schemas.jsonl golden files.")
    ap.add_argument("--out_dir", type=Path, default=Path("sft_data"))

    ap.add_argument("--metadata_dropout", type=float, default=0.0,
                    help="Probability to drop a metadata key (except language/source_modality).")

    ap.add_argument("--language", default="en")
    ap.add_argument("--na_token", default="")
    ap.add_argument("--source_modality", default="plain_text")

    ap.add_argument("--seed", type=int, default=0, help="Random seed for metadata dropout determinism.")

    args = ap.parse_args()
    random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for inp in args.inputs:
        outp = infer_out_path(inp, args.out_dir)
        convert_file(
            inp,
            outp,
            metadata_dropout=args.metadata_dropout,
            language=args.language,
            na_token=args.na_token,
            source_modality=args.source_modality,
        )
        print(f"[OK] wrote: {outp}")

if __name__ == "__main__":
    main()
