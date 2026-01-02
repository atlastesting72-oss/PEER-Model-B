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
# PEER Model - FIXED VERSION
# Changes:
#   1. Use FULL dataset (not 50%)
#   2. Increased training steps to 25,000
#   3. Fixed expert dimensions (proper mini-FFN with hidden dim)
#   4. Added learning rate warmup (1000 steps)
#   5. Added gradient clipping (max_norm=1.0)
#   6. Lowered learning rate to 1e-4
#   7. Separate logging for aux_loss
#   8. Increased context length to 512
# ==========================================

class PeerConfig:
    def __init__(self):
        # Model Dimensions
        self.vocab_size = 50257     # GPT-2 tokenizer
        self.d_model = 384          # Embedding size
        self.n_layer = 12           # Layers
        self.n_head = 6             # Heads
        self.max_len = 512          # Context window (INCREASED from 256)
        self.dropout = 0.1

        # Expert Config
        # 65,536 Experts per layer (256x256 grid)
        self.num_experts = 65536
        self.k_active = 16          # Active experts per head
        self.expert_dim = self.d_model // self.n_head  # Query dimension per head
        

        self.expert_hidden = 128    # Hidden dim inside each expert (was effectively just 64)

        # Training Settings
        self.batch_size = 32
        self.learning_rate = 1e-4   # LOWERED from 3e-4
        self.min_lr = 1e-5          # Minimum LR for cosine decay
        self.warmup_steps = 1000    # NEW: Warmup steps
        self.max_iters = 25000      # INCREASED from 5000
        self.eval_interval = 100
        self.grad_clip = 1.0        # NEW: Gradient clipping
        self.aux_loss_coef = 0.01   # Auxiliary loss coefficient
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = PeerConfig()
print(f"Device: {config.device}")
print(f"Config: d_model={config.d_model}, n_layer={config.n_layer}, n_head={config.n_head}")
print(f"Expert config: num_experts={config.num_experts}, k_active={config.k_active}, expert_hidden={config.expert_hidden}")

# ==========================================
# DATA LOADING - Using FULL dataset now
# ==========================================
print("Loading TinyStories (FULL dataset)...")
dataset = load_dataset("roneneldan/TinyStories", split="train")  
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def fast_tokenize(examples):

    out = tokenizer(examples["text"], truncation=False, add_special_tokens=False)

    input_ids = [ids + [tokenizer.eos_token_id] for ids in out["input_ids"]]
    return {"input_ids": input_ids}

print("Tokenizing data (multiprocessed)...")
tokenized = dataset.map(
    fast_tokenize,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=["text"]
)

block_size = config.max_len + 1

def group_texts(examples):

    concatenated = sum(examples["input_ids"], [])
    total_length = len(concatenated)

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size


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


lm_dataset.set_format(type="torch", columns=["input_ids"])

train_loader = DataLoader(
    lm_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True  
)

print(f"Dataset size: {len(lm_dataset)} chunks of {block_size} tokens")
print(f"Total training tokens available: {len(lm_dataset) * block_size / 1e6:.2f}M")

# ==========================================
# MODEL ARCHITECTURE - FIXED
# ==========================================

class ProductKeyRetrieval(nn.Module):
    """Product Key Memory for efficient expert routing."""
    def __init__(self, d_query, num_experts, k_active):
        super().__init__()
        self.k = k_active
        self.sqrt_n = int(math.sqrt(num_experts))
        self.sub_key_dim = d_query // 2

        self.c_keys = nn.Parameter(torch.randn(self.sqrt_n, self.sub_key_dim) * 0.02)
        self.c_prime_keys = nn.Parameter(torch.randn(self.sqrt_n, self.sub_key_dim) * 0.02)
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
            # Penalize concentration (lower entropy = higher penalty)
            aux_loss = (prob1.pow(2).sum() + prob2.pow(2).sum()) * self.sqrt_n

        return global_indices, final_scores, aux_loss


class PEERLayer(nn.Module):
    """
    PEER (Product of Experts with Extreme Retrieval) Layer.
    
    FIXED: Each expert is now a proper mini-FFN with hidden dimension,
    not just a pair of vectors.
    """
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.expert_hidden = config.expert_hidden

        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.router = ProductKeyRetrieval(self.head_dim, config.num_experts, config.k_active)

        self.w_down = nn.Embedding(config.num_experts, self.head_dim * self.expert_hidden)
        self.w_up = nn.Embedding(config.num_experts, self.expert_hidden * self.head_dim)

        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        nn.init.normal_(self.w_down.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.w_up.weight, mean=0.0, std=0.01)

    def forward(self, x):
        b, s, d = x.shape
        queries = self.query_proj(x).view(b, s, self.heads, self.head_dim)
        indices, scores, aux_loss = self.router(queries)


        k = indices.shape[-1]
        

        w_down_flat = self.w_down(indices)  # [b, s, h, k, head_dim * expert_hidden]
        w_down_vals = w_down_flat.view(b, s, self.heads, k, self.head_dim, self.expert_hidden)
        

        w_up_flat = self.w_up(indices)  # [b, s, h, k, expert_hidden * head_dim]
        w_up_vals = w_up_flat.view(b, s, self.heads, k, self.expert_hidden, self.head_dim)


        x_heads = x.view(b, s, self.heads, self.head_dim)
        

        hidden = torch.einsum('bshd,bshkde->bshke', x_heads, w_down_vals)
        hidden = F.gelu(hidden)

        routing_weights = F.softmax(scores, dim=-1)  # [b, s, h, k]
        hidden = hidden * routing_weights.unsqueeze(-1)  # [b, s, h, k, expert_hidden]


        out_heads = torch.einsum('bshke,bshked->bshd', hidden, w_up_vals)
        
        out = self.out_proj(out_heads.reshape(b, s, d))
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
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
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
            if aux is not None:
                total_aux_loss += aux
            new_kvs.append(kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        ce_loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            loss = ce_loss + self.config.aux_loss_coef * total_aux_loss

        return logits, loss, new_kvs, ce_loss, total_aux_loss


# ==========================================
# TRAINING SETUP
# ==========================================
print("Initializing model...")
model = PEERModel(config).to(config.device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Parameters: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable")

print("Compiling model (this takes a minute but speeds up training)...")
try:
    model = torch.compile(model)
    print("Model compiled successfully!")
except Exception as e:
    print(f"Warning: torch.compile failed ({e}). Proceeding without compilation.")


use_fused = config.device == 'cuda'
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, fused=use_fused, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler('cuda')


def get_lr(step):
    """Learning rate schedule with warmup and cosine decay."""

    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    decay_steps = config.max_iters - config.warmup_steps
    current_step = step - config.warmup_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / decay_steps))
    return config.min_lr + (config.learning_rate - config.min_lr) * cosine_decay


# ==========================================
# TRAINING LOOP
# ==========================================
print(f"\nStarting training for {config.max_iters} steps...")
print(f"Warmup: {config.warmup_steps} steps, LR: {config.learning_rate} -> {config.min_lr}")
print(f"Gradient clipping: {config.grad_clip}")
print("-" * 60)

start_time = time.time()
model.train()

step = 0
running_loss = 0.0
running_ce_loss = 0.0
running_aux_loss = 0.0
log_interval = 50

data_iter = iter(train_loader)

while step < config.max_iters:

    try:
        batch = next(data_iter)
    except StopIteration:
        print(f"[Epoch boundary at step {step}]")
        data_iter = iter(train_loader)
        batch = next(data_iter)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    data = batch["input_ids"].to(config.device, non_blocking=True)
    x, y = data[:, :-1], data[:, 1:]


    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, loss, _, ce_loss, aux_loss = model(x, targets=y)


    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    
    # Gradient clipping (FIXED: was missing!)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    
    scaler.step(optimizer)
    scaler.update()

    running_loss += loss.item()
    running_ce_loss += ce_loss.item()
    running_aux_loss += aux_loss if isinstance(aux_loss, float) else aux_loss.item()

    step += 1

    if step % log_interval == 0:
        avg_loss = running_loss / log_interval
        avg_ce = running_ce_loss / log_interval
        avg_aux = running_aux_loss / log_interval
        
        dt = time.time() - start_time
        tokens_per_sec = (config.batch_size * config.max_len * log_interval) / dt
        
        print(f"Step {step:5d}/{config.max_iters} | "
              f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Aux: {avg_aux:.4f}) | "
              f"LR: {lr:.2e} | "
              f"Speed: {tokens_per_sec/1000:.1f}k tok/s")
        
        running_loss = 0.0
        running_ce_loss = 0.0
        running_aux_loss = 0.0
        start_time = time.time()


    if step % 5000 == 0:
        checkpoint_path = f'peer_checkpoint_step{step}.pt'
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

print("\nTraining complete!")


@torch.no_grad()
def generate(prompt, max_tokens=100, temperature=0.7):
    model.eval()
    idx = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
    past_kv = None

    print(f"\nPrompt: {prompt}", end="")
    for _ in range(max_tokens):
        if past_kv is not None:
            input_idx = idx[:, -1:]
        else:
            input_idx = idx

        logits, _, past_kv, _, _ = model(input_idx, past_key_values=past_kv)

        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)
        word = tokenizer.decode(next_token[0])
        print(word, end="", flush=True)

        if next_token.item() == tokenizer.eos_token_id:
            break

    print("\n" + "-"*50)
    model.train()

print("\nGenerating Story Samples:")
generate("Once upon a time", max_tokens=150)
generate("The little girl found a", max_tokens=150)
generate("In a magical forest", max_tokens=150)


torch.save({
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
}, 'peer_model_final.pt')
print("Final model saved as peer_model_final.pt")
