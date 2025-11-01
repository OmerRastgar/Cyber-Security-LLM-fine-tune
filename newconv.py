import json

# Input JSONL file
input_file = "Dataset.jsonl"
# Output JSON file for Unsloth's load_dataset
output_file = "unsloth_ready_dataset.json"

all_conversations_data = []

print(f"Starting conversion from {input_file} to {output_file}...")

try:
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue  # skip empty lines
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line: {e}")
                continue

            system_prompt = data.get("system_prompt", "")
            user_prompt = data.get("user_prompt", "")
            ai_prompt = data.get("ai_prompt", "")
            document_name = data.get("document_name", "")

            # Merge the document name into the assistant's output
            merged_output = f"{ai_prompt} \n \n [Source: {document_name}] " if document_name else ai_prompt
            merged_output = merged_output.strip()

            # Build the list of conversation turns
            # This matches the ShareGPT format expected by the mapping:
            # {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}
            convo_turns = []
            
            if system_prompt:
                convo_turns.append({"from": "system", "value": system_prompt})
            
            if user_prompt:
                # 'human' maps to 'user' role in the template
                convo_turns.append({"from": "human", "value": user_prompt})
            
            if merged_output:
                # 'gpt' maps to 'assistant' role in the template
                convo_turns.append({"from": "gpt", "value": merged_output})

            # Only add if we have valid turns
            if convo_turns:
                # Each item in the list is an object with a "conversations" key
                all_conversations_data.append({"conversations": convo_turns})

    # Save the entire list as a single JSON file
    if all_conversations_data:
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(all_conversations_data, outfile, indent=4, ensure_ascii=False)
        print(f"✅ Conversion complete. {len(all_conversations_data)} examples saved to {output_file}")
    else:
        print("No valid data was found to convert.")

except FileNotFoundError:
    print(f"Error: The input file '{input_file}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

