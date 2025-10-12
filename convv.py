import json

# Input JSONL file
input_file = "fine_tune_dataset.csv.journal.jsonl"
# Output JSON file
output_file = "converted.json"

converted_data = []

with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        if not line.strip():
            continue  # skip empty lines
        data = json.loads(line)
        
        # Convert structure
        converted = {
            "output": data.get("ai_prompt", ""),
            "input": data.get("user_prompt", ""),
            "instruction": data.get("system_prompt", "")
        }
        converted_data.append(converted)

# Save as JSON
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(converted_data, outfile, indent=4, ensure_ascii=False)

print(f"✅ Conversion complete. JSON saved to {output_file}")
