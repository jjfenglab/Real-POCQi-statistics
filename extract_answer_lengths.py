#!/usr/bin/env python3
"""Extract per-answer text lengths from qa_answers_article_data.jsonl.

Reads <data_dir>/qa_answers_article_data.jsonl and writes
<data_dir>/answer_lengths.csv with columns: answer_id, answer_len.

answer_len is the character count of article_data.raw_text. Records
missing raw_text get an empty answer_len (read by R as NA).
"""
import csv
import json
import sys
from pathlib import Path


def main(data_dir: Path) -> None:
    src = data_dir / "qa_answers_article_data.jsonl"
    dst = data_dir / "answer_lengths.csv"

    n_rows = 0
    n_missing = 0
    with src.open() as fin, dst.open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["answer_id", "answer_len"])
        for line in fin:
            rec = json.loads(line)
            text = (rec.get("article_data") or {}).get("raw_text")
            if text is None:
                writer.writerow([rec["id"], ""])
                n_missing += 1
            else:
                writer.writerow([rec["id"], len(text)])
            n_rows += 1

    print(f"Wrote {n_rows} rows to {dst} ({n_missing} missing raw_text)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: extract_answer_lengths.py <data_dir>")
    main(Path(sys.argv[1]))
