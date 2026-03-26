import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import math

# --- 1. Dataset ---
# Small synthetic dataset for quick experimentation
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

# --- 3. Optimizers & Matrix Operations ---
# Base class for Muon-style optimization
class CustomMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ortho_method='polar_express'):
        defaults = dict(lr=lr, momentum=momentum, ortho_method=ortho_method)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # We only apply Muon to >=2D parameters (matrices)
                if p.ndim < 2:
                    # fallback to SGD with momentum for 1D params
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
                
                # Apply orthogonalization
                g = self.orthogonalize(g, method=group['ortho_method'])
                g = g.to(p.dtype)
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
                
    def orthogonalize(self, G, method):
        X = G.float()
        transpose_needed = G.size(-2) > G.size(-1)
        if transpose_needed: 
            X = X.mT 
            
        if method == 'polar_express':
            coeffs = [
                (8.156554524902461, -22.48329292557795, 15.878769915207462),
                (4.042929935166739, -2.808917465908714, 0.5000178451051316),
                (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
                (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
                (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
            ]
            X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
            for a, b, c in coeffs:
                A = X @ X.mT 
                A2 = A @ A 
                B = b * A + c * A2
                X = a * X + B @ X
                
        elif method == 'newton_schulz':
            X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
            for _ in range(5):
                A = X @ X.mT
                X = 1.5 * X - 0.5 * (A @ X)
                
        elif method == 'svd':
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            X = U @ Vh
            
        elif method == 'qr':
            # Not exactly orthogonal polar factor, but related
            # Using QR on transpose to get orthogonal rows if tall, but here we just do standard QR
            Q, R = torch.linalg.qr(X.mT)
            X = Q.mT
            
        elif method == 'sign':
            # Simple sign method
            X = torch.sign(X)

        if transpose_needed: 
            X = X.mT 
            
        return X

# --- 4. Training Loop ---
def run_experiment(name, optimizer_class, optim_kwargs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyLLM().to(device)
    
    # Param count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] Starting experiment (Params: {num_params/1000:.1f}K)")
    
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
        ppl = math.exp(min(l_val, 20)) # clip ppl for stability
        
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
        {'name': 'AdamW Baseline', 'class': torch.optim.AdamW, 'kwargs': {'lr': 0.001}},
        {'name': 'Muon (Polar Express)', 'class': CustomMuon, 'kwargs': {'lr': 0.02, 'ortho_method': 'polar_express'}},
        {'name': 'Muon (Newton-Schulz)', 'class': CustomMuon, 'kwargs': {'lr': 0.02, 'ortho_method': 'newton_schulz'}},
        {'name': 'Muon (SVD)', 'class': CustomMuon, 'kwargs': {'lr': 0.02, 'ortho_method': 'svd'}},
        {'name': 'Muon (QR)', 'class': CustomMuon, 'kwargs': {'lr': 0.02, 'ortho_method': 'qr'}},
        {'name': 'Muon (Sign)', 'class': CustomMuon, 'kwargs': {'lr': 0.02, 'ortho_method': 'sign'}}
    ]
    
    results = []
    for exp in experiments:
        res = run_experiment(exp['name'], exp['class'], exp['kwargs'])
        results.append(res)
        
    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    for res in results:
        # smooth loss
        s_loss = pd.Series(res['loss']).rolling(window=5, min_periods=1).mean()
        plt.plot(s_loss, label=res['name'])
    plt.title('Training Loss (Smoothed)')
    plt.xlabel('Step')
    plt.ylabel('Cross Entropy')
    plt.legend()
    plt.grid(True)
    
    # Perplexity plot
    plt.subplot(1, 2, 2)
    for res in results:
        s_ppl = pd.Series(res['ppl']).rolling(window=5, min_periods=1).mean()
        plt.plot(s_ppl, label=res['name'])
    plt.title('Perplexity (Smoothed)')
    plt.xlabel('Step')
    plt.ylabel('PPL')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiment_results.png', dpi=300)
    print("Saved plot to 'experiment_results.png'")
    
    # Save metrics to CSV
    df_metrics = pd.DataFrame()
    for res in results:
        df_metrics[res['name'] + '_loss'] = res['loss']
        df_metrics[res['name'] + '_ppl'] = res['ppl']
    
    df_metrics.to_csv('experiment_metrics_summary.csv', index=False)
    print("Saved metrics to 'experiment_metrics_summary.csv'")
    
if __name__ == "__main__":
    main()
