import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import math
import os
import time
import random

# ==========================================
# 1. CONFIGURATION
# ==========================================
class PeerConfig:
    def __init__(self):
        # Model Dimensions
        self.vocab_size = 50257     # GPT-2 tokenizer
        self.d_model = 384          # Embedding size

        # --- CHANGED: Increased from 8 to 12 ---
        self.n_layer = 12
        # ---------------------------------------

        self.n_head = 6             # Heads
        self.max_len = 256          # Context window
        self.dropout = 0.1

        # Expert Config
        # 65,536 Experts per layer (256x256 grid)
        self.num_experts = 65536
        self.k_active = 16          # Active experts per head
        self.expert_dim = self.d_model // self.n_head

        # Training Settings
        self.batch_size = 32
        self.learning_rate = 3e-4
        self.max_iters = 5000       # Steps to train
        self.eval_interval = 100
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = PeerConfig()
print(f"Device: {config.device}")

# ==========================================
# 2. OPTIMIZED DATA PIPELINE
# ==========================================
# Using all data, but streaming/mapping efficiently
print("Loading TinyStories...")
dataset = load_dataset("roneneldan/TinyStories", split="train[:50%]")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def fast_tokenize(examples):
    # Tokenize in batches using the Rust-based fast tokenizer
    out = tokenizer(examples["text"], truncation=False, add_special_tokens=False)
    # Add EOS token manually to every sequence
    input_ids = [ids + [tokenizer.eos_token_id] for ids in out["input_ids"]]
    return {"input_ids": input_ids}

print("Tokenizing data (multiprocessed)...")
# map() caches results on disk, so you only wait once.
tokenized = dataset.map(
    fast_tokenize,
    batched=True,
    num_proc=os.cpu_count(), # Use all CPU cores
    remove_columns=["text"]
)

# Flatten logic (simplified for speed)
# We pack tokens into fixed chunks of length 257 (256 context + 1 target)
block_size = config.max_len + 1

def group_texts(examples):
    # Concatenate all texts
    concatenated = sum(examples["input_ids"], [])
    total_length = len(concatenated)
    # Drop the small remainder
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size
    result = [
        concatenated[i : i + block_size]
        for i in range(0, total_length, block_size)
    ]
    return {"input_ids": result}

print("Grouping tokens into chunks...")
lm_dataset = tokenized.map(
    group_texts,
    batched=True,
    num_proc=os.cpu_count()
)

# Convert to PyTorch format without loading to RAM
lm_dataset.set_format(type="torch", columns=["input_ids"])

# High-performance DataLoader
train_loader = DataLoader(
    lm_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,   # Prepare data in background
    pin_memory=True  # Faster transfer to CUDA
)

# ==========================================
# 3. MODEL ARCHITECTURE (Production Ready)
# ==========================================

class ProductKeyRetrieval(nn.Module):
    def __init__(self, d_query, num_experts, k_active):
        super().__init__()
        self.k = k_active
        self.sqrt_n = int(math.sqrt(num_experts))
        self.sub_key_dim = d_query // 2

        self.c_keys = nn.Parameter(torch.randn(self.sqrt_n, self.sub_key_dim) * 0.1)
        self.c_prime_keys = nn.Parameter(torch.randn(self.sqrt_n, self.sub_key_dim) * 0.1)
        self.query_ln = nn.LayerNorm(d_query)

    def forward(self, query):
        b, s, h, d = query.shape
        q_norm = self.query_ln(query)
        q1 = q_norm[..., :self.sub_key_dim]
        q2 = q_norm[..., self.sub_key_dim:]

        scores1 = torch.matmul(q1, self.c_keys.t())
        scores2 = torch.matmul(q2, self.c_prime_keys.t())

        k_pre = min(self.k * 4, self.sqrt_n)
        top_s1, idx1 = torch.topk(scores1, k_pre, dim=-1)
        top_s2, idx2 = torch.topk(scores2, k_pre, dim=-1)

        joint_scores = top_s1.unsqueeze(-1) + top_s2.unsqueeze(-2)
        joint_flat = joint_scores.view(b, s, h, -1)
        final_scores, best_flat_idx = torch.topk(joint_flat, self.k, dim=-1)

        row_idx = best_flat_idx.div(k_pre, rounding_mode='floor')
        col_idx = best_flat_idx % k_pre
        real_row = torch.gather(idx1, -1, row_idx)
        real_col = torch.gather(idx2, -1, col_idx)
        global_indices = real_row * self.sqrt_n + real_col

        aux_loss = 0.0
        if self.training:
            prob1 = F.softmax(scores1, dim=-1).view(-1, self.sqrt_n).mean(0)
            prob2 = F.softmax(scores2, dim=-1).view(-1, self.sqrt_n).mean(0)
            aux_loss = (prob1.pow(2).sum() + prob2.pow(2).sum()) * self.sqrt_n

        return global_indices, final_scores, aux_loss

class PEERLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_head
        self.head_dim = config.d_model // config.n_head

        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.router = ProductKeyRetrieval(self.head_dim, config.num_experts, config.k_active)

        self.w_down = nn.Embedding(config.num_experts, self.head_dim)
        self.w_up = nn.Embedding(config.num_experts, self.head_dim)

        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        b, s, d = x.shape
        queries = self.query_proj(x).view(b, s, self.heads, self.head_dim)
        indices, scores, aux_loss = self.router(queries)

        w_down_vals = self.w_down(indices)
        w_up_vals = self.w_up(indices)

        x_heads = x.view(b, s, self.heads, 1, self.head_dim)
        hidden = (x_heads * w_down_vals).sum(dim=-1)
        hidden = F.gelu(hidden)

        routing_weights = F.softmax(scores, dim=-1)
        hidden = hidden * routing_weights

        out_heads = (hidden.unsqueeze(-1) * w_up_vals).sum(dim=3)
        out = self.out_proj(out_heads.view(b, s, d))
        return self.dropout(out), aux_loss

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.d_model
        self.head_dim = config.d_model // config.n_head
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x, past_kv=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        current_kv = (k, v)

        is_causal = True if past_kv is None else False
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=self.dropout if self.training else 0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), current_kv

class PEERBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.peer = PEERLayer(config)

    def forward(self, x, past_kv=None):
        attn_out, next_kv = self.attn(self.ln1(x), past_kv=past_kv)
        x = x + attn_out
        peer_out, aux_loss = self.peer(self.ln2(x))
        x = x + peer_out
        return x, aux_loss, next_kv

class PEERModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([PEERBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_key_values=None):
        device = idx.device
        b, t = idx.size()

        if past_key_values is None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
        else:
            past_length = past_key_values[0][0].size(2)
            pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)

        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)

        total_aux_loss = 0.0
        new_kvs = []

        for i, block in enumerate(self.blocks):
            last_kv = past_key_values[i] if past_key_values is not None else None
            x, aux, kv = block(x, past_kv=last_kv)
            if aux is not None: total_aux_loss += aux
            new_kvs.append(kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # FIX: Use .reshape() instead of .view() for targets.
            # 'targets' comes from a slice (data[:, 1:]), so it is non-contiguous in memory.
            # .view() requires contiguity, while .reshape() handles it automatically.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            loss = loss + 0.01 * total_aux_loss

        return logits, loss, new_kvs

# ==========================================
# 4. FAST TRAINING LOOP
# ==========================================
print("Initializing model...")
model = PEERModel(config).to(config.device)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# [OPTIMIZATION 1] Compile the model (Requires PyTorch 2.0+)
print("Compiling model (this takes a minute but speeds up training)...")
try:
    model = torch.compile(model)
except Exception as e:
    print(f"Warning: torch.compile failed ({e}). Proceeding without compilation.")

# [OPTIMIZATION 2] Fused Optimizer
# Check if fused is supported (usually needs CUDA)
use_fused = config.device == 'cuda'
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, fused=use_fused)
scaler = torch.amp.GradScaler('cuda')

# [OPTIMIZATION 3] Learning Rate Scheduler (Critical for convergence)
# Cosine decay helps the model settle faster
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_iters)

print("Starting training...")
start_time = time.time()
model.train()

step = 0
for batch in train_loader:
    # 1. Get Batch (already formatted)
    data = batch["input_ids"].to(config.device, non_blocking=True)
    x, y = data[:, :-1], data[:, 1:]

    # 2. Forward (Mixed Precision)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits, loss, _ = model(x, targets=y)

    # 3. Backward
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    step += 1

    # Logging
    if step % 50 == 0:
        dt = time.time() - start_time
        tokens_per_sec = (config.batch_size * config.max_len * 50) / dt
        print(f"Step {step}: Loss {loss.item():.4f} | Speed: {tokens_per_sec/1000:.1f}k tok/s")
        start_time = time.time()

    if step >= config.max_iters:
        break

print("Training complete!")

# ==========================================
# 5. GENERATION TEST
# ==========================================

@torch.no_grad()
def generate(prompt, max_tokens=100):
    model.eval()
    idx = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
    past_kv = None

    print(f"\nPrompt: {prompt}", end="")
    for _ in range(max_tokens):
        if past_kv is not None:
            input_idx = idx[:, -1:]
        else:
            input_idx = idx

        # Handle compiled model output variability (sometimes returns tensor or tuple)
        # However, our class definition returns (logits, loss, kvs)
        logits, _, past_kv = model(input_idx, past_key_values=past_kv)

        probs = F.softmax(logits[:, -1, :] / 0.7, dim=-1) # Temp 0.7 for stories
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)
        word = tokenizer.decode(next_token[0])
        print(word, end="")

        if next_token.item() == tokenizer.eos_token_id:
            break

    print("\n" + "-"*50)

print("\nGenerating Story Samples:")
generate("Once upon a time", max_tokens=150)
generate("The little girl found a", max_tokens=150)
torch.save(model.state_dict(), 'peer_opti_5k_step.pt')
print("Model saved as peer_opti_5k_step.pt")
