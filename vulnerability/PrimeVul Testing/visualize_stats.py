import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def visualize_dataset_stats(jsonl_path):
    labels = []
    token_lengths = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip(): continue
            record = json.loads(line)
            
            code = record.get('func', '')
            target = int(record.get('target', 0))
            
            if code:
                labels.append(target)
                tokens = tokenizer.encode(code, truncation=False)
                token_lengths.append(len(tokens))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x=labels, palette="Set2")
    plt.title("Class Distribution (0=Secure, 1=Vulnerable)")
    plt.xlabel("Target Label")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    sns.histplot(token_lengths, bins=50, kde=True, color="blue")
    plt.axvline(x=512, color='red', linestyle='--', label='512 Token Cutoff')
    plt.title("Code Token Length Distribution")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig("dataset_statistics.png", dpi=300)
    print("Saved visualization to 'dataset_statistics.png'")

if __name__ == "__main__":
    visualize_dataset_stats("PrimeVul_Data/primevul_train_paired.jsonl")