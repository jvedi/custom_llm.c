import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')

torch.save(model.state_dict(), 'gpt2_124M.bin')
