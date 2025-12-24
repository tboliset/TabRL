# TabRL-Summarizer

Two-stage text-to-table generation with LLMs and a MONA-style multi-policy reinforcement learning blueprint.

This repository accompanies the project **“TabRL-Summarizer: Two-Stage Text-to-Table Generation with MONA Multi-Policy Reinforcement Learning”** by  
**Arya Patil, Thanishka Bolisetty, Simran Sharma, Hriday Pradhan, Mohammed Usman Mehboob**  
**Arizona State University, Tempe, USA**

---

## Table of Contents

- [Overview](#overview)
- [Method](#method)
  - [Stage 1 – Schema Policy](#stage-1--schema-policy)
  - [Stage 2 – Table Policy](#stage-2--table-policy)
  - [MONA Multi-Policy RL (Blueprint)](#mona-multi-policy-rl-blueprint)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quickstart: End-to-End Demo](#quickstart-end-to-end-demo)
- [Training](#training)
  - [Data](#data)
  - [Stage 1 – Schema SFT](#stage-1--schema-sft)
  - [Stage 2 – Table SFT](#stage-2--table-sft)
- [Experiments](#experiments)
  - [LiveSum](#livesum)
  - [Rotowire](#rotowire)
- [Limitations & Future Work](#limitations--future-work)
- [Project Documents](#project-documents)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

Most analytics tools prefer **relational tables**, but information in the wild is written as **unstructured text**: sports recaps, biographies, financial reports, scientific articles, and more.

**TabRL-Summarizer** is a text-to-table system that:

1. Reads a document and predicts an explicit **schema** (column names, types, and structure).
2. Uses that schema to generate a **faithful tabular representation** of the document as JSONL (one JSON object per row).
3. Provides a **MONA-style** (Multi-Objective Non-Aggregating) RL blueprint that treats schema and table policies as separate agents with distinct rewards instead of one blended signal.

In short: the project turns messy narrative text into machine-readable tables and shows how to optimize the full text → schema → table pipeline.

---

## Method

High-level data flow:

```text
document x
   │
   ▼
Stage 1: schema policy π₁
   (LLM + LoRA)
   └─► schema s (JSON)
           │
           ▼
Stage 2: table policy π₂
   (LLM + LoRA)
   └─► table y (NDJSON)
