import torch
from model import TextHRM
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # just for tokenization

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_loader = DataLoader(tokenized_dataset["test"], batch_size=16)


# DataLoaders
train_loader = DataLoader(tokenized_dataset["train"].shuffle().select(range(1000)), batch_size=16, shuffle=True)
# test_loader = DataLoader(tokenized_dataset["test"].select(range(1000)), batch_size=16)


import torch.optim as optim

vocab_size = tokenizer.vocab_size
embed_dim = 128
hidden_dim = 256

# Training setup
model = TextHRM(L_cycles=4, H_cycles=2, vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, output_dim=1)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
device = torch.device("mps" if torch.mps.is_available() else "cpu")
model.to(device)


# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for i, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask).squeeze(-1)  # [batch_size]
        loss = criterion(outputs, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = (torch.sigmoid(outputs) > 0.5).long()
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

        if (i + 1) % 100 == 0:
            avg_loss = total_loss / (i + 1)
            accuracy = correct_predictions / total_predictions
            print(f"Epoch {epoch+1}, Batch {i+1}: Loss = {avg_loss:.4f}, Acc = {accuracy:.4f}")


    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

avg_train_loss = total_loss / len(train_loader)
train_accuracy = correct_predictions / total_predictions