import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    train_dataset = torch.load("dataset/train_dataset.pt", weights_only=False)
    valid_dataset = torch.load("dataset/valid_dataset.pt", weights_only=False)
    
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
    valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=16)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2
    )
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 3
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    print("Training Loop...")
    for epoch_i in range(epochs):
        print(f"\n Epoch {epoch_i + 1} / {epochs} ")
        total_train_loss = 0
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad()        
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            
            loss = loss_fn(outputs.logits, b_labels)
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            print(f"  Batch {step}  of  {len(train_dataloader)}. Loss: {loss.item():.4f}")
                
        avg_train_loss = total_train_loss / len(train_dataloader)            
        print(f"  Average training loss: {avg_train_loss:.4f}")
        
        # Validation Phase
        model.eval()
        total_eval_accuracy = 0
        
        for batch in valid_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():        
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            preds = np.argmax(logits, axis=1).flatten()
            total_eval_accuracy += np.sum(preds == label_ids) / len(label_ids)
            
        print(f"  Validation Accuracy: {total_eval_accuracy / len(valid_dataloader):.4f}")

    print("\nTraining complete.")
    model.save_pretrained("./codebert_model")

if __name__ == "__main__":
    train()