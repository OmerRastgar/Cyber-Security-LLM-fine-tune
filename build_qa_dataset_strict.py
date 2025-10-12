#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-style Q&A dataset builder (chunk-only, resumable, Ollama).

Per chunk:
- Generate 3 questions (detailed, medium, simple) strictly from that chunk.
- Answer each question strictly from the same chunk (length fits type).
- Save each Q/A as a separate CSV row.

Durability:
- CSV is created immediately (header) so it's visible while running.
- Each row is flushed and fsync'ed.
- A sidecar journal (.journal.jsonl) and checkpoint (.checkpoint.json) enable precise resume.
- If the main CSV is locked (e.g., open in Excel), rows go to a timestamped backup CSV.
"""

import os, re, csv, sys, json, time, hashlib, logging, argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

# Third-party: pip install requests pymupdf docx2txt tqdm
import requests
import fitz  # PyMuPDF
import docx2txt
from tqdm import tqdm

# -------------------- Text utils --------------------

def normalize_text(s: str) -> str:
    # join hyphenated line-breaks: authen-\ntication -> authentication
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    # collapse blank lines & spaces
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n"))
    return s.strip()

def is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".pdf", ".docx", ".txt"}

def read_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            parts = []
            with fitz.open(path) as doc:
                for p in doc:
                    parts.append(p.get_text("text"))
            return "\n".join(parts)
        if ext == ".docx":
            return docx2txt.process(path) or ""
        if ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        logging.exception("Read failed for %s: %s", path, e)
    return ""

def paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    paras = paragraphs(text)
    chunks, buf = [], ""
    for para in paras:
        candidate = (buf + ("\n\n" if buf else "") + para) if buf else para
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf: chunks.append(buf)
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    chunks.append(para[start:end])
                    start = max(end - overlap, end)
                buf = ""
            else:
                buf = para
    if buf: chunks.append(buf)
    if overlap > 0 and chunks:
        out = []
        for i, ch in enumerate(chunks):
            if i == 0:
                out.append(ch)
            else:
                tail = out[-1][-overlap:]
                merged = tail + ("\n\n" if tail and not tail.endswith("\n") else "") + ch
                out.append(merged[:chunk_size] if len(merged) > chunk_size else merged)
        chunks = out
    return chunks

# -------------------- Ollama --------------------

def ollama_generate(prompt: str, model: str, system: Optional[str], options: dict,
                    api_base: str = "http://localhost:11434", format_json: bool = False) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:  payload["system"] = system
    if options: payload["options"] = options
    if format_json: payload["format"] = "json"
    r = requests.post(f"{api_base}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response") or data.get("data") or ""

# -------------------- Prompts --------------------

QUESTION_SYSTEM = (
    "You generate three questions from the given chunk of a compliance/audit document.\n"
    "- Questions MUST be directly answerable using ONLY this chunk.\n"
    "- No organization-specific inventory (e.g., 'approved images'), vendor lists, or content not in the text.\n"
    "- Return STRICT JSON with keys: detailed, medium, simple."
)

QUESTION_USER_TMPL = """Document: {doc_name}

Chunk:
\"\"\"
{chunk}
\"\"\"

Create 3 questions derived strictly from this chunk:
- detailed: very specific, targets exact details/parameters/roles/frequency/sections present in the text.
- medium: summarizes the key requirement/idea of the chunk (still answerable from the text).
- simple: a small specific fact mentioned in the chunk.

Return STRICT JSON:
{{
  "detailed": "question ending with ?",
  "medium": "question ending with ?",
  "simple": "question ending with ?"
}}
"""

ANSWER_SYSTEM = (
    "You are a helpful assistant for cybersecurity audits. "
    "Answer ONLY from the provided chunk. "
    "If the answer is not present, reply exactly: 'Not in the provided context.' "
    "Include control identifiers/section numbers only if they appear in the chunk."
)

ANSWER_USER_TMPL = """Document: {doc_name}

Chunk:
\"\"\"
{chunk}
\"\"\"

Question ({qtype}):
{question}

Write an answer using ONLY the chunk above.
Style:
- detailed -> 1 short paragraph + bullet points as needed (150–250 words).
- medium   -> 3–6 sentences (80–150 words).
- simple   -> 1–3 sentences (30–70 words).

If the answer is not present in the chunk, reply exactly:
Not in the provided context.
"""

# -------------------- Row identity --------------------

@dataclass(frozen=True)
class RowKey:
    doc_name: str
    chunk_index: int
    qtype: str
    question: str
    def id(self) -> str:
        h = hashlib.sha256()
        parts = f"{self.doc_name}|{self.chunk_index}|{self.qtype}|{self.question}"
        h.update(parts.encode("utf-8"))
        return h.hexdigest()

# -------------------- Durable I/O --------------------

def _fsync(fh): fh.flush(); os.fsync(fh.fileno())

def ensure_csv(path: str, fieldnames: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL).writeheader()
            _fsync(f)

def append_journal(jpath: str, row: dict):
    os.makedirs(os.path.dirname(jpath) or ".", exist_ok=True)
    with open(jpath, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(row, ensure_ascii=False) + "\n"); _fsync(jf)

def append_csv(path: str, row: Dict[str, str], fieldnames: List[str]) -> Tuple[bool, Optional[str]]:
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            w.writerow(row); _fsync(f)
        return True, None
    except PermissionError:
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup = f"{path}.backup.{ts}.csv"
        logging.warning("CSV locked; writing to backup: %s", backup)
        exists = os.path.exists(backup)
        with open(backup, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            if not exists: w.writeheader()
            w.writerow(row); _fsync(f)
        return False, backup

def reconcile_journal(path: str, jpath: str, fieldnames: List[str]) -> int:
    if not os.path.exists(jpath): return 0
    existing = set()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    rid = r.get("row_id")
                    if rid: existing.add(rid)
        except Exception: pass
    appended = 0
    with open(jpath, "r", encoding="utf-8") as jf:
        for line in jf:
            line = line.strip()
            if not line: continue
            try: row = json.loads(line)
            except Exception: continue
            rid = row.get("row_id")
            if not rid or rid in existing: continue
            wrote, _ = append_csv(path, row, fieldnames)
            if wrote: existing.add(rid); appended += 1
    if appended:
        logging.info("Reconciled %d journal rows -> %s", appended, path)
    return appended

def load_seen(path: str, jpath: str) -> set:
    seen = set()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    rid = r.get("row_id"); 
                    if rid: seen.add(rid)
        except Exception: pass
    if os.path.exists(jpath):
        try:
            with open(jpath, "r", encoding="utf-8") as jf:
                for line in jf:
                    line = line.strip()
                    if not line: continue
                    try: row = json.loads(line)
                    except Exception: continue
                    rid = row.get("row_id"); 
                    if rid: seen.add(rid)
        except Exception: pass
    return seen

def save_checkpoint(cpath: str, data: dict):
    tmp = cpath + ".tmp"
    os.makedirs(os.path.dirname(cpath) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2); _fsync(f)
    os.replace(tmp, cpath)

# -------------------- Walk --------------------

def walk_files(d: str) -> List[str]:
    out = []
    for root, _, files in os.walk(d):
        for n in files:
            p = os.path.join(root, n)
            if is_supported(p): out.append(p)
    out.sort()
    return out

# -------------------- Core pipeline --------------------

def generate_three_questions(chunk: str, doc_name: str, model: str,
                             temperature: float, top_p: float, api_base: str) -> Dict[str, str]:
    user = QUESTION_USER_TMPL.format(doc_name=doc_name, chunk=chunk)
    out = ollama_generate(
        prompt=user, model=model, system=QUESTION_SYSTEM,
        options={"temperature": temperature, "top_p": top_p},
        api_base=api_base, format_json=True
    ).strip()
    # Parse strict JSON
    try:
        data = json.loads(out)
        qs = {
            "detailed": str(data.get("detailed", "")).strip(),
            "medium": str(data.get("medium", "")).strip(),
            "simple": str(data.get("simple", "")).strip(),
        }
    except Exception:
        # Fallback: try to extract a JSON object substring
        m = re.search(r"\{[\s\S]*\}", out)
        if m:
            try:
                data = json.loads(m.group(0))
                qs = {
                    "detailed": str(data.get("detailed", "")).strip(),
                    "medium": str(data.get("medium", "")).strip(),
                    "simple": str(data.get("simple", "")).strip(),
                }
            except Exception:
                qs = {"detailed":"", "medium":"", "simple":""}
        else:
            qs = {"detailed":"", "medium":"", "simple":""}

    # Final cleanup: must look like questions and be non-empty
    for k,v in list(qs.items()):
        if not v or not re.search(r"[?？]\s*$", v) or len(v) < 8:
            qs[k] = ""
    return qs

def answer_question(chunk: str, doc_name: str, qtype: str, question: str, model: str,
                    temperature: float, top_p: float, api_base: str) -> str:
    if not question:
        return ""
    user = ANSWER_USER_TMPL.format(doc_name=doc_name, chunk=chunk, qtype=qtype, question=question)
    # tune length via num_predict
    num_predict = 512 if qtype == "detailed" else (256 if qtype == "medium" else 128)
    out = ollama_generate(
        prompt=user, model=model, system=ANSWER_SYSTEM,
        options={"temperature": temperature, "top_p": top_p, "num_predict": num_predict},
        api_base=api_base, format_json=False
    ).strip()
    # Normalize exact "Not in the provided context." if model tried to decorate it
    nic = "Not in the provided context."
    if out.strip().lower().startswith("not in the provided context"):
        return nic
    return out

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Chunk-only three-question Q&A dataset builder (resumable)")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--model", default="qwen:14b")
    ap.add_argument("--questions_per_chunk", type=int, default=3, help="fixed to 3 (detailed, medium, simple)")
    ap.add_argument("--chunk_size", type=int, default=2000)
    ap.add_argument("--chunk_overlap", type=int, default=200)
    ap.add_argument("--temperature_q", type=float, default=0.2, help="temperature for question generation")
    ap.add_argument("--temperature_a", type=float, default=0.1, help="temperature for answering")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--api_base", default="http://localhost:11434")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint_path")
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--max_chunks_per_doc", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--drop_not_in_context", action="store_true",
                    help="If set, rows whose answer is 'Not in the provided context.' are NOT written.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    fieldnames = ["row_id","document_name","chunk_index","system_prompt","user_prompt","ai_prompt"]
    journal = args.output_csv + ".journal.jsonl"
    checkpoint = args.checkpoint_path or (args.output_csv + ".checkpoint.json")

    # Make CSV visible immediately
    ensure_csv(args.output_csv, fieldnames)
    logging.info("Output CSV: %s", os.path.abspath(args.output_csv))
    logging.info("Journal   : %s", os.path.abspath(journal))
    logging.info("Checkpoint: %s", os.path.abspath(checkpoint))

    # Resume
    if args.resume:
        reconcile_journal(args.output_csv, journal, fieldnames)
    seen = load_seen(args.output_csv, journal) if args.resume else set()

    files = walk_files(args.input_dir)
    if args.max_docs: files = files[:args.max_docs]
    logging.info("Found %d documents", len(files))

    # Load checkpoint
    start_doc_idx = 0
    start_chunk_map = {}
    if args.resume and os.path.exists(checkpoint):
        try:
            ck = json.load(open(checkpoint, "r", encoding="utf-8"))
            start_doc_idx = int(ck.get("doc_index", 0))
            start_chunk_map = ck.get("chunk_index_map", {})
            logging.info("Resuming from checkpoint: doc_index=%s", start_doc_idx)
        except Exception:
            logging.warning("Failed to parse checkpoint; starting fresh")

    rows_main = rows_backup = 0

    for di, path in enumerate(files, 0):
        if di < start_doc_idx: continue

        doc_name = os.path.basename(path)
        logging.info("Processing [%d/%d] %s", di + 1, len(files), doc_name)

        text = normalize_text(read_text(path))
        if not text or len(text) < 20:
            logging.warning("No extractable text in %s; skipping", doc_name)
            continue

        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        if args.max_chunks_per_doc:
            chunks = chunks[:args.max_chunks_per_doc]
        logging.info("Chunks: %d", len(chunks))

        # resume at chunk level
        start_chunk = 0
        if args.resume:
            try: start_chunk = int(start_chunk_map.get(doc_name, 0))
            except Exception: start_chunk = 0
        if start_chunk > 0:
            logging.info("Fast-forwarding to chunk %d for %s", start_chunk, doc_name)

        for ci in tqdm(range(start_chunk, len(chunks)), desc=f"Chunks for {doc_name}", unit="chunk"):
            chunk = chunks[ci]

            # 1) Generate three questions (detailed, medium, simple)
            qs = generate_three_questions(
                chunk, doc_name, args.model, args.temperature_q, args.top_p, args.api_base
            )

            # 2) Answer each, save each row
            for qtype in ("detailed", "medium", "simple"):
                q = qs.get(qtype, "")
                if not q:
                    continue
                rid = RowKey(doc_name, ci, qtype, q).id()
                if args.resume and rid in seen:
                    continue

                ans = answer_question(
                    chunk, doc_name, qtype, q, args.model, args.temperature_a, args.top_p, args.api_base
                )
                if args.drop_not_in_context and ans.strip() == "Not in the provided context.":
                    # Skip writing this row
                    save_checkpoint(checkpoint, {
                        "doc_index": di, "doc_name": doc_name,
                        "chunk_index_map": {**start_chunk_map, doc_name: ci},
                        "timestamp": time.time(),
                    })
                    continue

                row = {
                    "row_id": rid,
                    "document_name": doc_name,
                    "chunk_index": str(ci),
                    "system_prompt": ANSWER_SYSTEM,
                    "user_prompt": q,
                    "ai_prompt": ans,
                }

                append_journal(journal, row)
                wrote, backup = append_csv(args.output_csv, row, fieldnames)
                if wrote: rows_main += 1
                else: rows_backup += 1
                seen.add(rid)

                total = rows_main + rows_backup
                if total and (total % args.log_every == 0):
                    try: size_csv = os.path.getsize(args.output_csv)
                    except Exception: size_csv = 0
                    try: size_j = os.path.getsize(journal)
                    except Exception: size_j = 0
                    logging.info("Rows so far: main=%d, backup=%d | sizes csv=%dB journal=%dB",
                                 rows_main, rows_backup, size_csv, size_j)

                # checkpoint after each row
                save_checkpoint(checkpoint, {
                    "doc_index": di, "doc_name": doc_name,
                    "chunk_index_map": {**start_chunk_map, doc_name: ci},
                    "timestamp": time.time(),
                })

            # advance to next chunk
            save_checkpoint(checkpoint, {
                "doc_index": di, "doc_name": doc_name,
                "chunk_index_map": {**start_chunk_map, doc_name: ci + 1},
                "timestamp": time.time(),
            })

        # finished doc
        save_checkpoint(checkpoint, {
            "doc_index": di + 1, "doc_name": doc_name,
            "chunk_index_map": {**start_chunk_map, doc_name: len(chunks)},
            "timestamp": time.time(),
        })

    logging.info("Done. CSV: %s", os.path.abspath(args.output_csv))
    logging.info("Journal: %s", os.path.abspath(journal))
    logging.info("Checkpoint: %s", os.path.abspath(checkpoint))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr); sys.exit(130)
