import json

# Input JSONL file
input_file = "Dataset.jsonl"
# Output JSON file
output_file = "converted.json"

converted_data = []

with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        if not line.strip():
            continue  # skip empty lines
        data = json.loads(line)

        document_name = data.get("document_name", "")
        ai_prompt = data.get("ai_prompt", "")
        
        # Merge the document name into the output
        merged_output = f"{ai_prompt} \n \n [Source: {document_name}] " if document_name else ai_prompt

        converted = {
            "output": merged_output,
            "input": data.get("user_prompt", ""),
            "instruction": data.get("system_prompt", "")
        }
        converted_data.append(converted)

# Save as JSON
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(converted_data, outfile, indent=4, ensure_ascii=False)

print(f"✅ Conversion complete. JSON saved to {output_file}")
