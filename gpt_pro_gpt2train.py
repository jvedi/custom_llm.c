import torch
from torch.nn import Embedding, LayerNorm, Linear
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Example data: here you'd have your actual dataset
texts = ["Example sentence 1", "Example sentence 2"]
encoded_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
input_ids = encoded_inputs['input_ids'].to(device)
attention_mask = encoded_inputs['attention_mask'].to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # loop over the dataset multiple times
    optimizer.zero_grad()

    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save the model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
