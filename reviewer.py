import os, csv, time, argparse
from datetime import datetime
from typing import Optional, Set

import pandas as pd
import streamlit as st


def _rerun():
    import streamlit as st
    if hasattr(st, "rerun"):
        st.rerun()                  # new API
    else:
        st.experimental_rerun()     # old API

# ---------------- Args ----------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True, help="Path to your fine_tune_dataset.csv")
    p.add_argument("--label_csv", required=True, help="Where to save labels (appends)")
    p.add_argument("--chunk_size", type=int, default=500, help="Rows to load at a time")
    p.add_argument("--show_only_unlabeled", action="store_true", help="Hide already-labeled rows")
    # Parse only args after the -- that Streamlit forwards
    args, _ = p.parse_known_args()
    return args

# ---------------- File I/O helpers ----------------
NEEDED_COLS = ["row_id", "document_name", "chunk_index", "user_prompt", "ai_prompt"]

def count_rows_fast(path: str) -> int:
    # count lines minus header without loading to memory
    with open(path, "rb") as f:
        total = sum(1 for _ in f)
    return max(0, total - 1)

def ensure_label_file(path: str):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["row_id","decision","timestamp","document_name","chunk_index"])

def load_labeled_ids(path: str) -> Set[str]:
    s: Set[str] = set()
    if not os.path.exists(path):
        return s
    # stream in case it gets large
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = row.get("row_id")
            if rid:
                s.add(rid)
    return s

def append_label(path: str, row_id: str, decision: str, document_name: str, chunk_index: str):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow([row_id, decision, datetime.utcnow().isoformat(), document_name, chunk_index])
        f.flush()
        os.fsync(f.fileno())

def read_chunk(path: str, start_idx: int, chunk_size: int) -> pd.DataFrame:
    # Skip header + previous rows. For 15k rows this is totally fine.
    skiprows = range(1, start_idx + 1) if start_idx > 0 else None
    return pd.read_csv(
        path,
        skiprows=skiprows,
        nrows=chunk_size,
        usecols=NEEDED_COLS,
        dtype={"row_id": str, "document_name": str, "chunk_index": str, "user_prompt": str, "ai_prompt": str},
        encoding="utf-8"
    )

# ---------------- Session setup ----------------
def init_session(args):
    if "args" not in st.session_state:
        st.session_state.args = args
    if "total_rows" not in st.session_state:
        st.session_state.total_rows = count_rows_fast(args.input_csv)
    if "labeled_ids" not in st.session_state:
        ensure_label_file(args.label_csv)
        st.session_state.labeled_ids = load_labeled_ids(args.label_csv)
    if "cursor" not in st.session_state:
        st.session_state.cursor = 0  # global row index (excluding header)
    if "chunk_start" not in st.session_state:
        st.session_state.chunk_start = 0
    if "df_chunk" not in st.session_state:
        st.session_state.df_chunk = read_chunk(args.input_csv, 0, args.chunk_size)
    if "within_chunk_idx" not in st.session_state:
        st.session_state.within_chunk_idx = 0  # index inside the current chunk

def reload_chunk_if_needed():
    args = st.session_state.args
    cur = st.session_state.cursor
    start = st.session_state.chunk_start
    df = st.session_state.df_chunk

    if df is None or cur < start or cur >= start + len(df):
        # Load new chunk starting at current cursor
        st.session_state.chunk_start = cur
        st.session_state.df_chunk = read_chunk(args.input_csv, cur, args.chunk_size)
        st.session_state.within_chunk_idx = 0

def get_current_row(unlabeled_only: bool) -> Optional[pd.Series]:
    args = st.session_state.args
    total = st.session_state.total_rows
    cur = st.session_state.cursor

    while cur < total:
        reload_chunk_if_needed()
        start = st.session_state.chunk_start
        df = st.session_state.df_chunk

        if df is None or df.empty:
            # End of file
            return None

        local_idx = cur - start
        if local_idx >= len(df):
            # move to next row, which will trigger next chunk load
            st.session_state.cursor += 1
            cur = st.session_state.cursor
            continue

        row = df.iloc[local_idx]
        if unlabeled_only and str(row["row_id"]) in st.session_state.labeled_ids:
            st.session_state.cursor += 1
            cur = st.session_state.cursor
            continue

        # Found a row to present
        st.session_state.within_chunk_idx = local_idx
        return row

    return None

def advance_cursor():
    st.session_state.cursor += 1

# ---------------- UI ----------------
def main():
    args = get_args()
    st.set_page_config(page_title="Q/A CSV Reviewer", page_icon="✅", layout="centered")
    init_session(args)

    st.title("Q/A CSV Reviewer")
    st.caption(f"Input: `{args.input_csv}`  |  Labels: `{args.label_csv}`")

    colA, colB, colC, colD = st.columns([1,1,1,1])
    with colA:
        only_unlab = st.checkbox("Only unlabeled", value=args.show_only_unlabeled, help="Skip anything already labeled")
    with colB:
        st.write("")
        st.write(f"**Total rows:** {st.session_state.total_rows}")
    with colC:
        st.write("")
        st.write(f"**Labeled:** {len(st.session_state.labeled_ids)}")
    with colD:
        progress = 0.0
        if st.session_state.total_rows:
            progress = len(st.session_state.labeled_ids) / st.session_state.total_rows
        st.progress(progress, text=f"{int(progress*100)}% labeled")

    row = get_current_row(only_unlab)

    if row is None:
        st.success("🎉 Done! No more rows to review.")
        return

    # Present row
    st.markdown(f"**Document:** `{row['document_name']}` · **Chunk:** `{row['chunk_index']}`")
    st.markdown("### Question")
    st.markdown(f"> " + (row["user_prompt"] or "").strip())
    st.markdown("### Answer")
    st.markdown((row["ai_prompt"] or "").strip())

    # Action buttons
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        yes_clicked = st.button("✅ Yes (good for training)", use_container_width=True)
    with c2:
        no_clicked = st.button("❌ No (reject)", use_container_width=True)
    with c3:
        skip_clicked = st.button("⏭️ Skip", use_container_width=True)

    # Handle clicks
    if yes_clicked or no_clicked:
        decision = "yes" if yes_clicked else "no"
        append_label(
            args.label_csv,
            str(row["row_id"]),
            decision,
            str(row["document_name"]),
            str(row["chunk_index"]),
        )
        # update in-memory labeled set
        st.session_state.labeled_ids.add(str(row["row_id"]))
        advance_cursor()
        _rerun()
    elif skip_clicked:
        advance_cursor()
        _rerun()

    # Jump controls
    st.divider()
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        jump_to = st.number_input("Jump to global row index", min_value=0, max_value=max(0, st.session_state.total_rows-1),
                                  value=st.session_state.cursor, step=1)
    with col2:
        if st.button("Go", use_container_width=True):
            st.session_state.cursor = int(jump_to)
            _rerun()
    with col3:
        if st.button("Reload labels", use_container_width=True):
            st.session_state.labeled_ids = load_labeled_ids(args.label_csv)
            _rerun()

if __name__ == "__main__":
    main()
