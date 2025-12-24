#!/usr/bin/env python3
import json
import os
import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.cuda.empty_cache()

# ---------------------------------------------------------------------
#  Load RoBERTa-MNLI
# ---------------------------------------------------------------------
model_name = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ---------------------------------------------------------------------
#  Batched NLI with RoBERTa
# ---------------------------------------------------------------------
def nli_roberta_batch(pairs):
    """
    pairs: list of (premise, hypothesis)
    returns: list of entailment probabilities in [0, 1]
    """
    if not pairs:
        return []

    s1, s2 = zip(*pairs)
    inputs = tokenizer(
        list(s1),
        list(s2),
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).cpu().tolist()
    # label mapping: 0 = contradiction, 1 = neutral, 2 = entailment
    return [p[-1] for p in probs]


# ---------------------------------------------------------------------
#  PRF1 using NLI soft scores
# ---------------------------------------------------------------------
def precision_recall_f1(predicted_statements, gold_statements):
    N = len(predicted_statements)
    M = len(gold_statements)

    if N == 0 or M == 0:
        return 0.0, 0.0, 0.0

    # Precision: for each predicted, max entailment to any gold
    pred_gold_pairs = [(pi, gj) for pi in predicted_statements for gj in gold_statements]
    precision_ent_scores = nli_roberta_batch(pred_gold_pairs)

    precision_scores = []
    for i in range(N):
        start = i * M
        end = (i + 1) * M
        precision_scores.append(max(precision_ent_scores[start:end]))
    precision = sum(precision_scores) / N if N > 0 else 0.0

    # Recall: for each gold, max entailment to any predicted
    gold_pred_pairs = [(gj, pi) for gj in gold_statements for pi in predicted_statements]
    recall_ent_scores = nli_roberta_batch(gold_pred_pairs)

    recall_scores = []
    for j in range(M):
        start = j * N
        end = (j + 1) * N
        recall_scores.append(max(recall_ent_scores[start:end]))
    recall = sum(recall_scores) / M if M > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


# ---------------------------------------------------------------------
#  Parse tabunroll output → list of atomic statements
# ---------------------------------------------------------------------
def parse_atomic_statements(text):
    if not text:
        return []

    statements = []
    lines = text.splitlines()
    in_statements = False

    for line in lines:
        if line.strip().startswith("Statements:"):
            in_statements = True
            continue

        if in_statements:
            if not line.strip() or line.strip().startswith("Rows:"):
                break
            m = re.match(r"\d+\.\s*(.*)", line)
            if m:
                s = m.group(1).strip()
                if s:
                    statements.append(s)

    return statements


# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
def count_txt_files(path: str) -> int:
    if not path or not os.path.isdir(path):
        return 0
    return sum(
        1
        for f in os.listdir(path)
        if f.endswith(".txt") and os.path.isfile(os.path.join(path, f))
    )


def collect_ids_from_dir(path: str):
    if not path or not os.path.isdir(path):
        return set()
    files = [
        f for f in os.listdir(path)
        if f.endswith(".txt") and os.path.isfile(os.path.join(path, f))
    ]
    ids = set()
    for f in files:
        name, _ = os.path.splitext(f)
        ids.add(name)  # string ID
    return ids


def nice_name_from_dir(dirname: str) -> str:
    # "team" -> "Team", "player_stats" -> "Player Stats"
    name = dirname.replace("_", " ").strip()
    if not name:
        return dirname
    return " ".join(w.capitalize() for w in name.split())


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help=(
            "Predicted tabunrolled path.\n"
            "- If it has subdirs with .txt files (e.g. team/, player/), each is treated "
            "as a separate predicted table category.\n"
            "- If there are only .txt files in the root, the root is treated as a single "
            "category named 'Team' (for backward compatibility)."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output JSON file with entailment scores.",
    )
    parser.add_argument(
        "--gold_path",
        type=str,
        required=True,
        help="Gold tabunrolled path (gold files are <id>.txt directly in this dir).",
    )
    args = parser.parse_args()

    method_path = args.input_path
    eval_path = args.output_path
    gold_path = args.gold_path

    if not os.path.isdir(method_path):
        raise FileNotFoundError(f"Predicted path not found: {method_path}")
    if not os.path.isdir(gold_path):
        raise FileNotFoundError(f"Gold dir not found: {gold_path}")

    # ------------------ Discover predicted categories ------------------
    # 1) Subdirectories with txt files → categories
    category_dirs = {}  # {DisplayName: path}
    entries = os.listdir(method_path)

    for entry in entries:
        subdir = os.path.join(method_path, entry)
        if os.path.isdir(subdir) and count_txt_files(subdir) > 0:
            # Map 'team' -> 'Team', 'player' -> 'Player', etc.
            disp_name = entry
            # Preserve canonical 'Team'/'Player' naming for compat
            if entry.lower() == "team":
                disp_name = "Team"
            elif entry.lower() == "player":
                disp_name = "Player"
            else:
                disp_name = nice_name_from_dir(entry)
            category_dirs[disp_name] = subdir

    # 2) Root-level txt files: if no 'Team' category yet, treat root as Team
    root_has_txt = count_txt_files(method_path) > 0
    if root_has_txt and "Team" not in category_dirs:
        print(
            "[detect] No 'team/' subdir found; using root as 'Team' category "
            f"({method_path})."
        )
        category_dirs["Team"] = method_path
    elif root_has_txt:
        print(
            "[detect] Root has txt files but 'team/' subdir also exists; "
            "ignoring root-level files."
        )

    if not category_dirs:
        raise RuntimeError(
            f"No predicted .txt files found in {method_path} or its subdirectories."
        )

    print("[detect] Predicted categories and dirs:")
    for cname, cdir in category_dirs.items():
        print(f"  - {cname}: {cdir}")

    # ------------------ Gold IDs ------------------
    gold_ids = collect_ids_from_dir(gold_path)
    if not gold_ids:
        raise RuntimeError(f"No gold .txt files found in {gold_path}")

    # Predicted IDs union (across categories)
    pred_ids_union = set()
    for cdir in category_dirs.values():
        pred_ids_union |= collect_ids_from_dir(cdir)

    eval_ids = sorted(gold_ids & pred_ids_union)
    print(f"Found {len(eval_ids)} IDs in intersection of gold and any predicted files.")

    error_idx = []
    per_example = {}

    # sums for macro-averaging per category
    cat_sums = {cname: {"p": 0.0, "r": 0.0, "f1": 0.0} for cname in category_dirs}
    n_examples = 0

    # Cache gold statements per id to avoid rereading
    gold_cache = {}

    for idx in eval_ids:
        try:
            print("Idx:", idx)

            # GOLD
            if idx not in gold_cache:
                with open(os.path.join(gold_path, f"{idx}.txt"), "r") as f:
                    gold_text = f.read()
                gold_cache[idx] = parse_atomic_statements(gold_text)
            gold_statements = gold_cache[idx]

            per_example[idx] = {}
            n_examples += 1

            for cname, cdir in category_dirs.items():
                pred_file = os.path.join(cdir, f"{idx}.txt")

                if not os.path.exists(pred_file):
                    method_statements = []
                else:
                    with open(pred_file, "r") as f:
                        method_statements = parse_atomic_statements(f.read())

                if len(gold_statements) == 0 or len(method_statements) == 0:
                    p, r, f1 = 0.0, 0.0, 0.0
                else:
                    p, r, f1 = precision_recall_f1(method_statements, gold_statements)

                per_example[idx][cname] = {
                    "Precision": p,
                    "Recall": r,
                    "F1-score": f1,
                }

                cat_sums[cname]["p"] += p
                cat_sums[cname]["r"] += r
                cat_sums[cname]["f1"] += f1

        except Exception as e:
            print("Idx:", idx)
            print("Error:", e)
            error_idx.append(idx)

    # -----------------------------------------------------------------
    #  TabEval-style aggregate metrics (macro-averaged, in percent)
    # -----------------------------------------------------------------
    if n_examples > 0:
        aggregate_cats = {}
        for cname, sums in cat_sums.items():
            cp = (sums["p"] / n_examples) * 100.0
            cr = (sums["r"] / n_examples) * 100.0
            cf1 = (sums["f1"] / n_examples) * 100.0
            aggregate_cats[cname] = {
                "Correctness": cp,
                "Completeness": cr,
                "Overall": cf1,
            }
    else:
        aggregate_cats = {}

    # Backward-compatible top-level keys for Team/Player if they exist
    team_correctness = team_completeness = team_overall = 0.0
    player_correctness = player_completeness = player_overall = 0.0

    if "Team" in aggregate_cats:
        team_correctness = aggregate_cats["Team"]["Correctness"]
        team_completeness = aggregate_cats["Team"]["Completeness"]
        team_overall = aggregate_cats["Team"]["Overall"]

    if "Player" in aggregate_cats:
        player_correctness = aggregate_cats["Player"]["Correctness"]
        player_completeness = aggregate_cats["Player"]["Completeness"]
        player_overall = aggregate_cats["Player"]["Overall"]

    aggregate = {
        "num_examples": n_examples,
        "num_errors": len(error_idx),
        "categories": aggregate_cats,  # general form
    }

    # Also expose Team/Player if present (for old scripts)
    if "Team" in aggregate_cats:
        aggregate["Team"] = aggregate_cats["Team"]
    if "Player" in aggregate_cats:
        aggregate["Player"] = aggregate_cats["Player"]

    # Pretty print
    print("\n=== TabEval (macro-averaged, %) ===")
    for cname, metrics in aggregate_cats.items():
        print(
            f"{cname:10s} - Correctness: {metrics['Correctness']:6.2f}, "
            f"Completeness: {metrics['Completeness']:6.2f}, "
            f"Overall: {metrics['Overall']:6.2f}"
        )
    print(f"(Examples: {n_examples}, Errors: {len(error_idx)})")

    # -----------------------------------------------------------------
    #  Write JSON: aggregate + per-example
    # -----------------------------------------------------------------
    output = {
        "aggregate": aggregate,
        "per_example": per_example,
        "error_indices": error_idx,
    }

    with open(eval_path, "w") as f:
        json.dump(output, f, indent=6)

    print("\nSaved results to:", eval_path)
