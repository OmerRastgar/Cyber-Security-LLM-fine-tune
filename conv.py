#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a CSV produced by build_qa_dataset.py into an Unsloth-friendly dataset:
- Reads columns: system_prompt, user_prompt, ai_prompt (required)
- Emits a JSONL with a single key per row: {"messages": [ ... ]}
  where messages is a list of dicts with keys {"role", "content"}
- Optionally also writes a Parquet file via Hugging Face Datasets for faster loading

Usage:
  python csv_to_unsloth_messages.py \
    --input_csv ./fine_tune_dataset.csv \
    --output_jsonl ./compliance_messages.jsonl \
    --output_parquet ./compliance_messages.parquet

Then in Python:
  from datasets import load_dataset
  ds = load_dataset("json", data_files="./compliance_messages.jsonl", split="train")
  # or: ds = load_dataset("parquet", data_files="./compliance_messages.parquet", split="train")

  from unsloth.chat_templates import get_chat_template
  tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

  def to_text(batch):
      msgs_list = batch["messages"]
      texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
               for msgs in msgs_list]
      return {"text": texts}

  ds = ds.map(to_text, batched=True)
"""

import os
import csv
import json
import argparse

try:
    from datasets import Dataset
except Exception:
    Dataset = None


def row_to_messages(row: dict) -> list:
    msgs = []
    sys_prompt = (row.get("system_prompt") or "").strip()
    user_prompt = (row.get("user_prompt") or "").strip()
    ai_prompt = (row.get("ai_prompt") or "").strip()

    if sys_prompt:
        msgs.append({"role": "system", "content": sys_prompt})
    if user_prompt:
        msgs.append({"role": "user", "content": user_prompt})
    if ai_prompt:
        msgs.append({"role": "assistant", "content": ai_prompt})
    return msgs


def convert_csv(input_csv: str, output_jsonl: str, output_parquet: str = None):
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    # Stream JSONL writing for memory safety
    with open(input_csv, "r", encoding="utf-8", newline="") as f_in, \
         open(output_jsonl, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            msgs = row_to_messages(row)
            if not msgs:
                continue
            rec = {"messages": msgs}
            json.dump(rec, f_out, ensure_ascii=False)
            f_out.write("\n")

    if output_parquet:
        if Dataset is None:
            raise SystemExit("Install `datasets` to write parquet: pip install datasets pyarrow")
        from datasets import load_dataset
        ds = load_dataset("json", data_files=output_jsonl, split="train")
        ds.to_parquet(output_parquet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--output_parquet")
    args = ap.parse_args()

    convert_csv(args.input_csv, args.output_jsonl, args.output_parquet)
    print(f"Wrote: {args.output_jsonl}")
    if args.output_parquet:
        print(f"Wrote: {args.output_parquet}")

if __name__ == "__main__":
    main()
