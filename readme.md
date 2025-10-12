# Compliance Q&A Dataset Builder (Ollama + LLaMA)

This tool turns your compliance documents (PDF/DOCX/TXT) into a fine-tuning CSV by:
1. Extracting and chunking text with overlap.
2. Asking an Ollama-served model to generate audit-style questions per chunk.
3. Asking the model to answer each question using **only** that chunk.
4. Appending rows to a CSV with fields: `document_name, chunk_index, system_prompt, user_prompt, ai_prompt`.

It **saves after every row** to be crash-safe and can **resume** without duplications.

## Quick Start

1) Install dependencies:
```bash
pip install -r requirements.txt
Run Ollama locally and pull a model:

bash
Copy code
ollama serve
ollama pull llama3.1
Run the builder:

bash
Copy code
python data.py --input_dir ./my_docs --output_csv ./fine_tune_dataset.csv --model qwen:14b --questions_per_chunk 3 --chunk_size 2000 --chunk_overlap 200 --resume
Convert CSV → Unsloth messages
bash
Copy code
python conv.py --input_csv ./fine_tune_dataset.csv --output_jsonl ./compliance_messages.jsonl --output_parquet ./compliance_messages.parquet
Then in Python:

python
Copy code
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

ds = load_dataset("json", data_files="compliance_messages.jsonl", split="train")
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

def to_text(batch):
    return {"text": [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in batch["messages"]
    ]}

ds = ds.map(to_text, batched=True)
yaml
Copy code

---

If you’d like, I can also paste a tiny **one-shot shell script** that writes all four files to disk automatically (useful if copy/paste is finicky).
::contentReference[oaicite:0]{index=0}