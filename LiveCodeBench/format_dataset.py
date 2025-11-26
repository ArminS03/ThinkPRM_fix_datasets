import re
import json


def extract_python_code(text):
    pattern = r'```python\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""


def process_jsonl(input_file):
    completions = {2**i: [] for i in range(0, 6)}
    
    with open(input_file, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            
            for i in range(0, 6):
                key = f"completion_weighted@{2**i}"
                if key in data:
                    completions[2**i].append({
                        "question_id": data["question_id"],
                        "code_list": [extract_python_code(data[key])]
                    })
    
    for n in range(0, 6):
        key = 2**n
        output_filename = f"./dataset/completions_{key}.json"
        
        with open(output_filename, 'w') as out_file:
            json.dump(completions[key], out_file, indent=4)

        print(f"Generated {output_filename}")

input_file = '/fs/nexus-scratch/armins/ThinkPRM/outputs/results/test-time-scaling/livecodebench/Qwen2.5-Coder-7B-Instruct/best_of_n_disc_prm_qwen/best_of_n_completions.jsonl'
process_jsonl(input_file)
