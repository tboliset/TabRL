# TabRL two-stage text-to-table generation

TabRL separates table generation into two inspectable steps: predict a typed JSON schema from source text, then generate a table conditioned on the text and schema. The project contains Llama 3.1 8B supervised fine-tuning workflows, LoRA checkpoints, stored predictions and a separate T3-style atomic-statement evaluation pipeline.

The MONA-style reinforcement-learning extension is documented as a design. It is not a completed RL training result or evidence of improvement from RL.

## Start with the evidence

- [Final team report](docs/Stratos_Final_Report.pdf): method, experiments and team contributions.
- [Schema checkpoint](https://huggingface.co/mohdusman001/schema-gen-llama3-8b-stage1-merged) and [table checkpoint](https://huggingface.co/mohdusman001/pi2-table-llama3-8b-sft_final).
- [Stored schema predictions](src/schema_model/SFT_output/testing_schema_table_schemas_pred.jsonl) and [stored table predictions](src/table_model/SFT_output/testing_schema_table_tables_pred.jsonl): 120 records each for error analysis, with separate malformed-output files.
- [Testing inputs](examples/testing_input/testing_schema_table.jsonl): 120 text/schema/table records. These counts describe artifacts, not a quality score or a new evaluation.

## Repository layout

```text
TabRL/
├── docs/
│   ├── Stratos_Final_Report.pdf
│   ├── Work_Till_Now_NLP.pdf
│   └── RL-pipeline.pdf
├── examples/
│   ├── sample_input/sample_data.jsonl
│   └── testing_input/testing_schema_table.jsonl
├── src/
│   ├── schema_model/
│   │   ├── schema_instructions.py
│   │   ├── schema_train_sft.py
│   │   └── SFT_output/
│   ├── table_model/
│   │   ├── table_instructions.py
│   │   ├── table_train_sft.py
│   │   └── SFT_output/
│   └── common/eval/
│       ├── map&make/
│       │   ├── atomization.py
│       │   ├── schema_generator.py
│       │   └── table_generator.py
│       └── tab_eval/
│           ├── tabunroll_infer.py
│           └── entailment_eval.py
└── README.md
```

The trainers are the substantive scripts under `src/schema_model/` and `src/table_model/`, not wrappers around missing root-level files. There is no committed `src/RL/` implementation directory; the RL design is in the PDFs.

## Inspect without training or model downloads

From the repository root:

```bash
python3 src/schema_model/schema_instructions.py --help
python3 src/table_model/table_instructions.py --help
python3 src/schema_model/schema_train_sft.py --help
python3 src/table_model/table_train_sft.py --help
```

These four entry points parse help before training dependencies or model loading. They are a small way to inspect the actual command-line interfaces, not an end-to-end training test. The committed sample has 18 records; it is an input fixture, not a complete training dataset.

The GPU training and inference commands below require a separately configured compatible environment and access to the selected base model. This repository does not provide a pinned environment lockfile. The training scripts check PyTorch, VERL, FlashAttention-2, PEFT, Transformers, Datasets and PyArrow; the atomic-statement inference scripts additionally import vLLM. Match library and CUDA versions to the target machine. Documentation path checks do not establish a newly reproduced training run.

## Prepare supervised instruction data

The committed input examples use JSONL records with `id`, `text`, `schema` and `table`. The schema and table builders have different output-directory flag spellings:

```bash
python3 src/schema_model/schema_instructions.py \
  --inputs examples/sample_input/sample_data.jsonl \
  --out_dir outputs/schema_messages

python3 src/table_model/table_instructions.py \
  --inputs examples/sample_input/sample_data.jsonl \
  --out-dir outputs/table_messages
```

The schema builder constructs system/user/assistant messages for minimal schema induction. It skips examples without text, schema or columns. The table builder validates required fields and schema/table column consistency before producing messages whose assistant target is NDJSON. Use actual training and validation data prepared with the same builders for a training run; do not treat the small shared sample as an independent evaluation split.

## Train the schema and table models

Both trainers expose LoRA settings, staged context lengths, learning rates, batch settings, checkpointing and time budgets. The source defaults progress through 2,048, 4,096 and 8,192 tokens. `--max_steps` caps steps **per context stage**, not the entire three-stage run.

Examples for a configured CUDA training environment:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 src/schema_model/schema_train_sft.py \
  --train_file /path/to/schema_train_messages.jsonl \
  --eval_file /path/to/schema_validation_messages.jsonl \
  --output_dir /path/to/checkpoints/schema_sft \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max_steps 200

CUDA_VISIBLE_DEVICES=0,1 python3 src/table_model/table_train_sft.py \
  --train_file /path/to/table_train_messages.jsonl \
  --eval_file /path/to/table_validation_messages.jsonl \
  --output_dir /path/to/checkpoints/table_sft \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max_steps 200
```

The scripts prepare Parquet data, launch VERL's FSDP SFT trainer and support checkpoint merging and sample generation. Source defaults are not a complete provenance record for every published checkpoint. Consult the report and checkpoint-specific records before attributing an exact training configuration or result.

## Atomic-statement evaluation pipeline

The separate map-and-make pipeline uses vLLM to move from text to atomic statements, then a schema, then a table. These are base-model inference examples, not calls to the two fine-tuned checkpoints. The `map&make` paths must be quoted in a shell, and the base-model flag is `--model_name`.

```bash
mkdir -p outputs

python3 'src/common/eval/map&make/atomization.py' \
  --input examples/sample_input/sample_data.jsonl \
  --output outputs/atomized.jsonl \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct

python3 'src/common/eval/map&make/schema_generator.py' \
  --input outputs/atomized.jsonl \
  --output outputs/schema.jsonl \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct

python3 'src/common/eval/map&make/table_generator.py' \
  --input outputs/schema.jsonl \
  --output outputs/tables.jsonl \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct
```

Atomization accepts `text` or `summary`. The later scripts consume the generated atomic-statement and schema fields. Other argument spellings also differ between scripts, so inspect their argument definitions before adapting batch or GPU settings. These examples have been checked against file paths and argument definitions, not newly executed with a model.

[Table unrolling](src/common/eval/tab_eval/tabunroll_infer.py) converts tables back to atomic statements. [Entailment evaluation](src/common/eval/tab_eval/entailment_eval.py) uses RoBERTa-large-MNLI. The entailment script loads its model at module import, so importing it or asking for help can initiate model loading; it is not part of the no-download inspection above.

## Results and limitations

Read the report with its named baselines and evaluation conditions. Stored outputs include malformed generations so failures can be examined. A prompting comparison in that report is not an RL gain, and the available results do not establish uniformly best or state-of-the-art quality.

The RL proposal treats schema and table policies separately, with structure/validity and table-quality rewards. Its design is described in [RL-pipeline.pdf](docs/RL-pipeline.pdf) and the [final report](docs/Stratos_Final_Report.pdf); no completed RL experiment is claimed here.

## Team and contributions

Team: Arya Patil, Thanishka Bolisetty, Simran Sharma, Hriday Pradhan and Usman Mehboob Mohammed (listed as Mohammed Usman Mehboob in the original report).

The final report credits Usman with system design, both SFT pipelines, data preparation, training runs and integration. See the report for the full team contribution split.

For the original project narrative, see [Work_Till_Now_NLP.pdf](docs/Work_Till_Now_NLP.pdf).

## Citation

```bibtex
@misc{tabrl-summarizer,
  title = {TabRL-Summarizer: Two-Stage Text-to-Table Generation with MONA Multi-Policy Reinforcement Learning},
  author = {Patil, Arya and Bolisetty, Thanishka and Sharma, Simran and Pradhan, Hriday and Mehboob, Mohammed Usman},
  howpublished = {\url{https://github.com/tboliset/TabRL}},
  year = {2025}
}
```

The project title retains the proposed RL direction; cite the report's specific completed methods and results rather than inferring an RL result from the title.

Contact: [musmanme@asu.edu](mailto:musmanme@asu.edu).
