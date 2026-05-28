import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL_CONFIG


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0))

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        return (
            self.cos_cached[:, :seq_len, :],
            self.sin_cached[:, :seq_len, :],
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(x, seq_len=S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))

        seq_len_q = q.shape[2]
        seq_len_k = k.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=x.device, dtype=torch.bool),
            diagonal=seq_len_k - seq_len_q + 1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        attn_output = self.out_proj(attn_output)

        return attn_output, new_kv


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff_norm = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        residual = x
        x = self.attn_norm(x)
        x, new_kv = self.attn(x, attention_mask, past_kv, use_cache)
        x = residual + x

        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = residual + x

        return x, new_kv


class GPTModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        self.config = config

        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.dropout = nn.Dropout(config["dropout"])
        self.layers = nn.ModuleList([
            TransformerBlock(
                config["d_model"],
                config["n_heads"],
                config["d_ff"],
                config["dropout"],
            )
            for _ in range(config["n_layers"])
        ])
        self.final_norm = RMSNorm(config["d_model"])
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)

        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=False):
        B, S = input_ids.shape

        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        new_kvs = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_kv = layer(x, attention_mask, past_kv, use_cache)
            if use_cache:
                new_kvs.append(new_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss, "past_key_values": new_kvs}

    @torch.no_grad()
    def generate(self, input_ids, tokenizer, max_new_tokens=128, temperature=0.1,
                 top_p=0.9, repetition_penalty=1.2, attention_mask=None):
        self.eval()
        device = next(self.parameters()).device

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)

        past_kv = None
        generated = input_ids

        for _ in range(max_new_tokens):
            if past_kv is None:
                outputs = self(generated, attention_mask=attention_mask, use_cache=True)
            else:
                last_token = generated[:, -1:]
                outputs = self(last_token, past_key_values=past_kv, use_cache=True)

            logits = outputs["logits"][:, -1, :]
            past_kv = outputs["past_key_values"]

            if repetition_penalty > 1.0:
                for token_id in generated[0].tolist():
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)

                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
                sorted_probs[sorted_indices_to_remove] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()

                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        return generated[0].tolist()

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
