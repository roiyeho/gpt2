import torch
import torch.nn as nn
from transformer import TokenEmbedding, PositionalEmbedding, TransformerBlock

class GPT2(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads, hidden_dim, num_layers):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEmbedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, x, mask=None):
        seq_length = x.shape[1]
        mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        x = self.token_emb(x) + self.pos_emb(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.lm_head(x)
