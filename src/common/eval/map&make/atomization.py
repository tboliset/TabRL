#!/usr/bin/env python3
"""
Atomize summaries using Llama 3.1 8B Instruct with vLLM.

- Uses vLLM's LLM engine for fast batched inference.
- Supports single- and multi-GPU via --tensor-parallel-size and CUDA_VISIBLE_DEVICES.
- Uses a HuggingFace tokenizer + chat template to build proper Llama-3.1 prompts.
- Reads and writes JSONL.
- Allows controlling vLLM GPU memory usage via --gpu-memory-utilization.

Expected input (JSONL, one object per line), e.g.:
{"id": 0, "summary": "...", "table": {...}}

Output (JSONL):
{
  "id": 0,
  "summary": "...",
  "table": {...},
  "raw_output": "<full model text>",
  "atomic_statements": [
    "The Atlanta Hawks' record is 46 wins.",
    ...
  ]
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


ATOMIZATION_PROMPT = """You are an expert at converting unstructured, detailed textual inputs into highly structured and organized atomic statements.

***TASK***: 
Decompose the given paragraphs or sentences into clear, self-contained, and highly detailed short atomic statements without losing any information. Each atomic statement should capture a single fact or action with maximum granularity.

***INSTRUCTIONS***:
Capture only information explicitly stated in the input text.
No detail should be assumed, inferred, or added that is not present in the text.
Each atomic statement should contain only one key entity and one action or attribute.
If a sentence contains multiple pieces of information, decompose it further into more granular statements.
Eliminate ambiguity by resolving pronouns and ensuring each statement stands alone.
Preserve necessary context so each statement is meaningful on its own.
Represent numerical data and units exactly as given in the input text.
Ensure each statement conveys unique information without overlapping others.
Ensure statements are clear, direct, and free from unnecessary complexity.
DO NOT infer or calculate additional data points, such as missed shots, unless explicitly stated in the text.
Resolve pronouns to their corresponding nouns for clarity.
Maintain relationships between entities without combining multiple facts.

***OUTPUT FORMAT***:
<REASONING STEPS>

### Atomic Statements:
<ATOMIC STATEMENT 1>
<ATOMIC STATEMENT 2>
<ATOMIC STATEMENT 3>
...


***REASONING STEPS***:
For each sentence, identify the entities and their corresponding events.

***FINAL CHECKLIST***:
All information from the input text is included.
No information or calculation is added that is not present in the text.
Every fact and detail is accurately represented.
Statements are clear and can be understood independently.
Numerical data and units are formatted exactly as provided in the text.
Each statement directly reflects the input text without inferred details.
Pronouns are resolved; statements are unambiguous.
Each statement contains only one key entity and one action or attribute.

Do not number the statements or add extra formatting.
Provide the *OUTPUT* with *REASONING STEPS* in the specified format only.
"""


def setup_hf_cache(hf_home: Optional[str]):
    """
    Configure Hugging Face cache directories under a scratch path.
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

    - tensor_parallel_size controls how many GPUs to shard across.
      Make sure CUDA_VISIBLE_DEVICES is set appropriately.
    - dtype is bfloat16 by default for H100s / modern GPUs.
    - gpu_memory_utilization controls the fraction of each GPU vLLM tries to use.
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


def build_chat_prompt(tokenizer, text: str) -> str:
    """
    Build a Llama-3.1-style chat prompt using the system atomization instructions
    and the user's input text.
    """
    messages = [
        {"role": "system", "content": ATOMIZATION_PROMPT},
        {
            "role": "user",
            "content": f"Input text:\n{text}\n",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def parse_atomic_statements(raw_output: str) -> List[str]:
    """
    Extract atomic statements from the model's response.

    We look for the section after "### Atomic Statements:" and split line by line,
    dropping empty lines and any tags.
    """
    marker = "### Atomic Statements:"
    if marker not in raw_output:
        # Fallback: return all non-empty lines
        lines = [l.strip() for l in raw_output.splitlines() if l.strip()]
        return lines

    after = raw_output.split(marker, 1)[1]
    lines = [l.strip() for l in after.splitlines()]

    atomic: List[str] = []
    for line in lines:
        if not line:
            continue
        # stop if we hit another section marker
        if line.startswith("***") or line.startswith("<REASONING STEPS>"):
            break
        atomic.append(line)

    return atomic


def generate_batch(
    llm: LLM,
    tokenizer,
    records: List[Dict[str, Any]],
    sampling_params: SamplingParams,
) -> List[str]:
    """
    Run generation for a batch of summaries with vLLM.
    Each record in `records` must have a 'text' key.
    """
    prompts = [build_chat_prompt(tokenizer, r["text"]) for r in records]

    # vLLM's generate handles multiple prompts efficiently
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    completions: List[str] = []
    for i, out in enumerate(outputs):
        if not out.outputs:
            raise RuntimeError(f"No outputs returned for record index {i}")
        completions.append(out.outputs[0].text.strip())

    return completions


def atomize_file(
    input_path: Path,
    output_path: Path,
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
    batch_size: int,
    log_interval: int,
):
    """
    Read JSONL from input_path, run atomization in batches, and write JSONL to output_path.
    Progress is logged after each batch and every `log_interval` records.
    """
    print(
        f"[atomize_file] Starting atomization\n"
        f"  input:  {input_path}\n"
        f"  output: {output_path}\n"
        f"  batch_size: {batch_size}\n"
        f"  log_interval: {log_interval}",
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
            text = obj.get("summary") or obj.get("text")
            if text is None:
                raise ValueError(
                    f"Line {line_idx}: no 'summary' or 'text' field found."
                )

            record = {
                "id": obj.get("id", line_idx),
                "summary": text,
                "table": obj.get("table"),
                "text": text,
                "line_idx": line_idx,
            }
            batch.append(record)

            # If batch filled up, run generation
            if len(batch) >= batch_size:
                print(
                    f"[atomize_file] Generating for batch "
                    f"{total_processed}–{total_processed + len(batch) - 1} "
                    f"(size={len(batch)})...",
                    flush=True,
                )
                completions = generate_batch(
                    llm=llm,
                    tokenizer=tokenizer,
                    records=batch,
                    sampling_params=sampling_params,
                )

                for rec, completion in zip(batch, completions):
                    atomic_statements = parse_atomic_statements(completion)
                    out_obj = {
                        "id": rec["id"],
                        "summary": rec["summary"],
                        "table": rec["table"],
                        "raw_output": completion,
                        "atomic_statements": atomic_statements,
                    }
                    f_out.write(json.dumps(out_obj) + "\n")

                f_out.flush()
                total_processed += len(batch)

                print(
                    f"[atomize_file] Completed batch. "
                    f"Total processed so far: {total_processed}",
                    flush=True,
                )

                if total_processed % log_interval == 0:
                    print(
                        f"[atomize_file] PROGRESS: {total_processed} records processed.",
                        flush=True,
                    )

                batch = []

        # Handle final partial batch
        if batch:
            print(
                f"[atomize_file] Generating final batch "
                f"{total_processed}–{total_processed + len(batch) - 1} "
                f"(size={len(batch)})...",
                flush=True,
            )
            completions = generate_batch(
                llm=llm,
                tokenizer=tokenizer,
                records=batch,
                sampling_params=sampling_params,
            )

            for rec, completion in zip(batch, completions):
                atomic_statements = parse_atomic_statements(completion)
                out_obj = {
                    "id": rec["id"],
                    "summary": rec["summary"],
                    "table": rec["table"],
                    "raw_output": completion,
                    "atomic_statements": atomic_statements,
                }
                f_out.write(json.dumps(out_obj) + "\n")

            f_out.flush()
            total_processed += len(batch)
            print(
                f"[atomize_file] Final batch done. Total processed: {total_processed}",
                flush=True,
            )

    print("[atomize_file] Atomization complete ✅", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file with 'summary' or 'text' field per line.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file with atomization results.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--hf_home",
        type=str,
        default=None,
        help="Base directory for Hugging Face caches (e.g., /scratch/$USER/hf_home).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
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
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to shard the model across (vLLM tensor_parallel_size).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help=(
            "Fraction of each GPU memory vLLM may use (0.0–1.0). "
            "If you see 'Free memory on device ... is less than desired', "
            "lower this value."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of records to process per vLLM batch (higher = faster, more VRAM).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log a progress message every N records.",
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
        f"[main] GPU mem utilization: {args.gpu_memory_utilization}",
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

    atomize_file(
        input_path=input_path,
        output_path=output_path,
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
