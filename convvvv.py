import json

with open("converted.jsonl", "r", encoding="utf-8") as f, open("clean.jsonl", "w", encoding="utf-8") as out:
    for line in f:
        try:
            json.loads(line)  # validate
            out.write(line)
        except:
            continue  # skip bad lines

print("Cleaned JSONL saved as clean.jsonl ✅")