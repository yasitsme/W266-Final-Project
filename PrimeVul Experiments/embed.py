import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.to(device)
model.eval()

def extract_embeddings(jsonl_path, sample_size=500):
    embeddings = []
    labels = []
    
    print(f"Extracting base embeddings for {sample_size} samples...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:sample_size]
        
        for line in tqdm(lines):
            if not line.strip(): continue
            record = json.loads(line)
            
            code = record.get('func', '')
            target = int(record.get('target', 0))
            
            if code:
                # Tokenize sequence
                inputs = tokenizer(code, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Extract  hidden states
                with torch.no_grad():
                    outputs = model(**inputs)
                    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                    
                embeddings.append(cls_embedding)
                labels.append(target)
                
    return np.array(embeddings), np.array(labels)

def plot_tsne(embeddings, labels):
    print("t-SNE dimensionality reduction ...")
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_embeddings[:, 0], 
        y=reduced_embeddings[:, 1], 
        hue=["Vulnerable" if l == 1 else "Secure" for l in labels],
        palette={"Secure": "green", "Vulnerable": "red"},
        alpha=0.7,
        s=60
    )

    plt.title("CodeBERT Embeddings (Before Fine-Tuning)", fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Class")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("tsne_before_training.png", dpi=300)
    print("Saved semantic visualization to 'tsne_before_training.png'")

if __name__ == "__main__":
    emb, lbls = extract_embeddings("PrimeVul_Data/primevul_train_paired.jsonl", sample_size=500)
    plot_tsne(emb, lbls)