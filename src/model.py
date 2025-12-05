import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

# --- Configuration Object ---
@dataclass
class ModelArgs:
    """
    Hyperparameters for the Omni-LLM (Mini-Llama Architecture).
    """
    dim: int = 256              # Embedding dimension
    n_layers: int = 4           # Transformer blocks
    n_heads: int = 8            # Query heads
    n_kv_heads: int = 4         # Grouped Query Attention (GQA) heads
    vocab_size: int = 50257     # GPT-2 tokenizer compatible
    multiple_of: int = 32       # SwiGLU hidden layer multiple
    max_seq_len: int = 512      # Context Window
    dropout: float = 0.1
    lora_rank: int = 4          # Rank for LoRA adaptation

# --- 1. RMSNorm (Root Mean Square Normalization) ---
class RMSNorm(nn.Module):
    """
    Llama uses RMSNorm instead of LayerNorm. 
    It is computationally cheaper (no mean subtraction) and more stable.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# --- 2. LoRA Layer (Parameter Efficient Fine-Tuning) ---
class LoRALayer(nn.Module):
    """
    Implements Low-Rank Adaptation (LoRA).
    Instead of full fine-tuning, we train two small matrices A and B.
    Formula: W_new = W + (A @ B) * scaling
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, alpha: int = 16):
        super().__init__()
        self.std_layer = nn.Linear(in_dim, out_dim, bias=False)
        
        # LoRA specific parameters
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank
        self.active = False # Switch to toggle 'Fine-Tuned' mode
        
        # Initialize LoRA weights
        # A=Gaussian (Random), B=Zeros ensures the layer acts like the base layer at start
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.std_layer(x)
        if self.active:
            lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
            return base_out + lora_out
        return base_out

# --- 3. Attention Mechanism (GQA + KV-Cache) ---
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        # Use LoRA for Query and Value projections
        self.wq = LoRALayer(args.dim, args.n_heads * self.head_dim, rank=args.lora_rank)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = LoRALayer(args.dim, self.n_kv_heads * self.head_dim, rank=args.lora_rank)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, past_kv: Optional[Tuple] = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape for multi-head
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # --- KV Cache Logic ---
        if past_kv is not None:
            xk_past, xv_past = past_kv
            xk = torch.cat([xk_past, xk], dim=1)
            xv = torch.cat([xv_past, xv], dim=1)
        
        current_kv = (xk, xv) # Save for next step

        # Grouped Query Attention: Repeat K/V heads to match Q heads
        xk = torch.repeat_interleave(xk, self.n_heads // self.n_kv_heads, dim=2)
        xv = torch.repeat_interleave(xv, self.n_heads // self.n_kv_heads, dim=2)

        # Transpose for Attention Calculation
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, xv)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output), current_kv, scores

# --- 4. SwiGLU Feed Forward ---
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # Round up to multiple_of for hardware efficiency
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# --- 5. Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x, freqs_cis, past_kv=None):
        # Pre-Norm Architecture
        h_att, kv, scores = self.attention(self.attention_norm(x), freqs_cis, past_kv)
        h = x + h_att
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, kv, scores

# --- 6. The Omni-Llama Model ---
class OmniLlama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.token_embedding = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        h = self.token_embedding(idx)
        freqs_cis = None 
        all_scores = []
        
        for layer in self.layers:
            h, _, scores = layer(h, freqs_cis, past_kv=None)
            all_scores.append(scores)
            
        h = self.norm(h)
        logits = self.output(h)
        return logits, all_scores

    def toggle_lora(self, active: bool):
        """
        Dynamically enable/disable LoRA adapters.
        """
        for module in self.modules():
            if isinstance(module, LoRALayer):
                module.active = active

# --- 7. Sanity Check ---
if __name__ == "__main__":
    print("Initializing Omni-Llama...")
    args = ModelArgs()
    model = OmniLlama(args)
    print(f"Model Architecture: Llama-3-Nano")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Created a dummy input (Batch Size 1, Sequence Length 10)
    dummy_input = torch.randint(0, args.vocab_size, (1, 10))
    logits, _ = model(dummy_input)
    print(f"Output Shape: {logits.shape}")
    print("Test Passed: The Brain is functioning.")