import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import math

# --- 1. Dataset ---
vocab_size = 256
seq_len = 32
batch_size = 32
num_steps = 150

def get_batch():
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda' if torch.cuda.is_available() else 'cpu')
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda' if torch.cuda.is_available() else 'cpu')
    return x, y

# --- 2. Minimal LLM (< 1M params) ---
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffp(self.ln2(x))
        return x

class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 64
        self.n_heads = 2
        self.n_layers = 2
        
        self.emb = nn.Embedding(vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(seq_len, self.d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(self.d_model, self.n_heads) for _ in range(self.n_layers)])
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, vocab_size, bias=False)
        
    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)
        x = self.emb(x) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

# --- 3. Optimizers & Matrix Operations (Iteration 2) ---
class CustomMuonIter2(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ortho_method='newton_schulz_5'):
        defaults = dict(lr=lr, momentum=momentum, ortho_method=ortho_method)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                if p.ndim < 2:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)
                    buf = state["momentum_buffer"]
                    buf.lerp_(p.grad, 1 - group["momentum"])
                    p.grad.lerp_(buf, group["momentum"])
                    p.add_(p.grad, alpha=-group["lr"])
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"])
                
                g = self.orthogonalize(g, method=group['ortho_method'])
                g = g.to(p.dtype)
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
                
    def orthogonalize(self, G, method):
        X = G.float()
        transpose_needed = G.size(-2) > G.size(-1)
        if transpose_needed: 
            X = X.mT 
            
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
        
        if method == 'newton_schulz_3':
            for _ in range(3):
                A = X @ X.mT
                X = 1.5 * X - 0.5 * (A @ X)
                
        elif method == 'newton_schulz_5':
            for _ in range(5):
                A = X @ X.mT
                X = 1.5 * X - 0.5 * (A @ X)
                
        elif method == 'newton_schulz_7':
            for _ in range(7):
                A = X @ X.mT
                X = 1.5 * X - 0.5 * (A @ X)
                
        elif method == 'svd_newton_hybrid':
            # 1 step of SVD approx, then newton
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            X = U @ Vh
            for _ in range(2):
                A = X @ X.mT
                X = 1.5 * X - 0.5 * (A @ X)

        if transpose_needed: 
            X = X.mT 
            
        return X

# --- 4. Training Loop ---
def run_experiment(name, optimizer_class, optim_kwargs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyLLM().to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] Starting Iteration 2 (Params: {num_params/1000:.1f}K)")
    
    optimizer = optimizer_class(model.parameters(), **optim_kwargs)
    
    losses = []
    perplexities = []
    
    start_time = time.time()
    for step in range(num_steps):
        x, y = get_batch()
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        l_val = loss.item()
        ppl = math.exp(min(l_val, 20)) 
        
        losses.append(l_val)
        perplexities.append(ppl)
        
        if step % 50 == 0:
            print(f"  Step {step}: Loss = {l_val:.4f}, PPL = {ppl:.2f}")
            
    dur = time.time() - start_time
    print(f"[{name}] Finished in {dur:.2f}s | Final Loss: {losses[-1]:.4f}\n")
    
    return {
        'name': name,
        'loss': losses,
        'ppl': perplexities,
        'time': dur
    }

def main():
    experiments = [
        {'name': 'NS-5 Baseline (from Iter1)', 'class': CustomMuonIter2, 'kwargs': {'lr': 0.02, 'ortho_method': 'newton_schulz_5'}},
        {'name': 'NS-3 Steps', 'class': CustomMuonIter2, 'kwargs': {'lr': 0.02, 'ortho_method': 'newton_schulz_3'}},
        {'name': 'NS-7 Steps', 'class': CustomMuonIter2, 'kwargs': {'lr': 0.02, 'ortho_method': 'newton_schulz_7'}},
        {'name': 'Hybrid SVD-NS', 'class': CustomMuonIter2, 'kwargs': {'lr': 0.02, 'ortho_method': 'svd_newton_hybrid'}},
        {'name': 'NS-5 (High Momentum 0.99)', 'class': CustomMuonIter2, 'kwargs': {'lr': 0.02, 'momentum': 0.99, 'ortho_method': 'newton_schulz_5'}}
    ]
    
    results = []
    for exp in experiments:
        res = run_experiment(exp['name'], exp['class'], exp['kwargs'])
        results.append(res)
        
    # --- Plotting Iteration 2 ---
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    for res in results:
        s_loss = pd.Series(res['loss']).rolling(window=5, min_periods=1).mean()
        plt.plot(s_loss, label=res['name'])
    plt.title('Iter 2: Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Cross Entropy')
    plt.legend()
    plt.grid(True)
    
    # Perplexity plot
    plt.subplot(1, 2, 2)
    for res in results:
        s_ppl = pd.Series(res['ppl']).rolling(window=5, min_periods=1).mean()
        plt.plot(s_ppl, label=res['name'])
    plt.title('Iter 2: Perplexity')
    plt.xlabel('Step')
    plt.ylabel('PPL')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiment_results_iter2.png', dpi=300)
    print("Saved plot to 'experiment_results_iter2.png'")
    
    df_metrics = pd.DataFrame()
    for res in results:
        df_metrics[res['name'] + '_loss'] = res['loss']
        df_metrics[res['name'] + '_ppl'] = res['ppl']
    
    df_metrics.to_csv('experiment_metrics_iter2.csv', index=False)
    print("Saved metrics to 'experiment_metrics_iter2.csv'")

if __name__ == "__main__":
    main()
