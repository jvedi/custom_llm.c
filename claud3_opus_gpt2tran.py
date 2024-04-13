import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# GPT-2模型配置
class GPT2Config:
    def __init__(self, max_seq_len, vocab_size, num_layers, num_heads, channels):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.channels = channels

# GPT-2模型定义
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Parameter(torch.zeros(config.vocab_size, config.channels))
        self.wpe = nn.Parameter(torch.zeros(config.max_seq_len, config.channels))

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        self.ln_f = nn.LayerNorm(config.channels)

    def forward(self, inp, targets=None):
        B, T = inp.shape
        C = self.config.channels
        V = self.config.vocab_size

        # encoder
        tok_emb = F.embedding(inp, self.wte) # B, T, C
        pos_emb = F.embedding(torch.arange(T, device=inp.device), self.wpe) # B, T, C
        x = tok_emb + pos_emb # B, T, C

        # transformer blocks
        for block in self.blocks:
            x = block(x)

        # LM head
        x = self.ln_f(x)
        logits = torch.matmul(x, self.wte.t()) # B, T, V

        if targets is not None:
            # 计算loss
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        else:
            loss = None

        return logits, loss

# 单个Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.channels)
        self.ln2 = nn.LayerNorm(config.channels)

        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# 注意力机制
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        C = config.channels
        self.C = C
        self.num_heads = config.num_heads
        self.head_size = C // self.num_heads

        self.qkv_proj = nn.Linear(C, 3*C)
        self.out_proj = nn.Linear(C, C)

        self.register_buffer("mask", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)))
        self.scale = 1 / math.sqrt(self.head_size)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x) # B, T, 3C
        qkv = qkv.view(B, T, self.num_heads, 3*self.head_size)
        qkv = qkv.permute(0, 2, 1, 3) # B, H, T, 3C
        q, k, v = qkv.chunk(3, dim=-1) # B, H, T, C

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale # B, H, T, T
        atty = torch.matmul(attn_weights.masked_fill(self.mask[:T,:T]==0, float('-inf')).softmax(-1), v)
        atty = atty.transpose(1, 2) # B, T, H, C
        atty = atty.reshape(B, T, self.C) # B, T, C

        out = self.out_proj(atty)
        return out

# MLP层
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        C = config.channels

        self.fc1 = nn.Linear(C, 4*C)
        self.fc2 = nn.Linear(4*C, C)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    # 从checkpoint加载模型参数
def load_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)

# 使用模型生成文本
def generate(model, start_tokens, max_length):
    B, T = start_tokens.shape
    generated = start_tokens
    model.eval()

    for _ in range(max_length):
        logits, _ = model(generated)
        probs = F.softmax(logits[:,-1,:], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    return generated

# 训练主循环
def train(model, optimizer, train_data, epochs, batch_size, seq_len, device):
    model.train()

    for epoch in range(epochs):
        for i in range(0, train_data.size(0)-1, batch_size):
            optimizer.zero_grad()

            batch = train_data[i:i+batch_size]
            inputs = batch[:, :seq_len]
            targets = batch[:, 1:seq_len+1]

            logits, loss = model(inputs.to(device), targets.to(device))

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

        samples = generate(model, torch.zeros((1, 1), dtype=torch.long, device=device), 64)
        print("Generated:", samples[0].tolist())

if __name__ == "__main__":
    # 配置
    config = GPT2Config(
        max_seq_len = 1024,
        vocab_size = 50257,
        num_layers = 12,
        num_heads = 12,
        channels = 768
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT2(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

    # 加载预训练权重
    load_model(model, 'gpt2_124M.bin')

    # 准备数据
    train_data = torch.randint(0, config.vocab_size, size=(100, config.max_seq_len))

    # 训练
    train(model, optimizer, train_data, epochs=10, batch_size=4, seq_len=64, device=device)