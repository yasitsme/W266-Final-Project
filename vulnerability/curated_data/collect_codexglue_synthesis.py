import json
from datasets import load_dataset
from tqdm import tqdm
import ollama

def generate_description(code_snippet, label):
    prompt = f"""You are a C/C++ security expert.

    Analyze the following C/C++ code snippet.

    Return exactly one sentence that identifies the primary vulnerability or security risk.
    Use a specific vulnerability name whenever possible (e.g., buffer overflow, integer overflow, use-after-free, null pointer dereference, out-of-bounds read, unchecked return value, race condition, format string vulnerability).

    If no vulnerability is present, reply exactly: No obvious vulnerability detected.

    Code:
    {code_snippet}

    Answer:
    """

    try:
        response = ollama.generate(
            model='qwen2.5-coder:7b',
            prompt=prompt,
            options={'temperature': 0.1, 'top_p': 0.9, 'num_predict': 30}
        )
        description = response['response'].strip().replace('\n', ' ')
        
        if not description or len(description) < 10:
            description = "This function contains a security vulnerability due to improper memory handling, logic flow, or unsafe function usage."
        return description
    except Exception as e:
        return "This function contains a structural security defect that creates an exploitable flow condition."

def main():
    print("Step 1: Downloading Microsoft CodeXGLUE Defect Detection split...")
    raw_dataset = load_dataset("google/code_x_glue_cc_defect_detection", split="train")
    
    output_path = "dataset.jsonl"
    print(f"Step 2: Synthesizing dataset and writing to {output_path}...")
    
    max_samples = 4000 
    sample_count = 0
    
    with open(output_path, "w", encoding="utf-8") as jsonl_file:
        for idx, item in enumerate(tqdm(raw_dataset, total=max_samples)):
            if sample_count >= max_samples:
                break
                
            record = dict(item) 
            code_snippet = str(record.get('func', ''))
            target_label = int(record.get('target', 0))
            
            reason_sentence = generate_description(code_snippet, target_label)
            
            formatted_record = {
                "func": code_snippet,
                "name": f"codexglue_idx_{idx}",
                "label": str(target_label),
                "cwe_id": "None", 
                "reason": reason_sentence
            }
            
            jsonl_file.write(json.dumps(formatted_record, ensure_ascii=False) + "\n")
            sample_count += 1
            

    print(f"\nComplete: Dataset with {sample_count} samples.")

if __name__ == '__main__':
    main()


# deepseek attempt
# import json
# import ollama
# from tqdm import tqdm
# from datasets import load_dataset
# import re


# def build_prompt(code_snippet):
#     return f"""
#         You are a C/C++ security expert.

#         Analyze the code and extract security reasoning.

#         Return ONLY valid JSON:

#         {{
#         "source": "what introduces the vulnerability (or none)",
#         "sink": "where the vulnerability triggers (or none)",
#         "cwe": "CWE-ID (e.g., CWE-119) or none",
#         "reason": "one clear sentence explanation"
#         }}

#         Rules:
#         - If no vulnerability exists:
#         source = "none"
#         sink = "none"
#         cwe = "none"
#         reason = "secure implementation"

#         - Do NOT include extra text.

#         Code:
#         {code_snippet}
#     """

# def safe_parse(output):
#     try:
#         return json.loads(output)
#     except:
#         match = re.search(r"\{.*\}", output, re.S)
#         if match:
#             try:
#                 return json.loads(match.group())
#             except:
#                 return None
#         return None


# def generate_reasoning(code_snippet):
#     prompt = build_prompt(code_snippet)

#     for _ in range(3):  
#         try:
#             response = ollama.generate(
#                 model="deepseek-r1:8b",
#                 prompt=prompt,
#                 options={
#                     "temperature": 0.1,
#                     "top_p": 0.9,
#                     "num_predict": 120
#                 }
#             )

#             parsed = safe_parse(response["response"])
#             if parsed:
#                 return parsed

#         except Exception:
#             continue

#     return {
#         "source": "none",
#         "sink": "none",
#         "cwe": "none",
#         "reason": "invalid or failed analysis"
#     }


# def is_valid(parsed):
#     if not parsed:
#         return False
#     if "reason" not in parsed:
#         return False
#     if len(parsed["reason"].split()) < 2:
#         return False
#     return True


# def main():
#     print("Loading CodeXGLUE dataset...")
#     dataset = load_dataset(
#         "google/code_x_glue_cc_defect_detection",
#         split="train"
#     )

#     output_file = "data.jsonl"
#     max_samples = 4000

#     count = 0

#     with open(output_file, "w", encoding="utf-8") as f:
#         for i, item in enumerate(tqdm(dataset, total=max_samples)):

#             if count >= max_samples:
#                 break

#             record = dict(item) 
#             code = str(record.get('func', ''))
#             label = int(record.get('target', 0))

#             parsed = generate_reasoning(code)

#             if not is_valid(parsed):
#                 continue

#             record = {
#                 "id": f"codex_{i}",
#                 "func": code,
#                 "label": label,

#                 "source": parsed["source"],
#                 "sink": parsed["sink"],
#                 "cwe": parsed["cwe"],
#                 "reason": parsed["reason"],

#             }
#             if label == 1:
#                 record["positive_text"] = parsed["reason"]
#                 record["negative_text"] = "secure function with no vulnerability"
#             else:
#                 record["positive_text"] = "secure function with no vulnerability"
#                 record["negative_text"] = parsed["reason"]

#             f.write(json.dumps(record) + "\n")
#             count += 1

#     print(f"Done. Saved {count} samples → {output_file}")


# if __name__ == "__main__":
#     main()