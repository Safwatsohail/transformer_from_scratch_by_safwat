Here’s a single complete, clean Transformer (paper-style) in one piece — encoder + decoder + masks + training-ready forward. No extra explanation, no split files.

import torch
import torch.nn as nn
import math
# -------------------- Attention --------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.q(q).view(B, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.k(k).view(B, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.v(v).view(B, -1, self.nhead, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out(x)
# -------------------- FFN --------------------
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)
# -------------------- Positional Encoding --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
# -------------------- Encoder --------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ffn(x))
        return x
# -------------------- Decoder --------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self, x, enc, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, enc, enc, src_mask))
        x = self.norm3(x + self.ffn(x))
        return x
# -------------------- Transformer --------------------
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=512,
        nhead=8,
        num_layers=6,
        d_ff=2048,
        max_len=5000
    ):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff)
            for _ in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, tgt_vocab)
    def tgt_mask(self, n, device):
        return torch.tril(torch.ones(n, n, device=device)).bool()
    def forward(self, src, tgt):
        B, Tsrc = src.shape
        B, Ttgt = tgt.shape
        src = self.src_emb(src) * math.sqrt(self.d_model)
        tgt = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        src = self.pos(src)
        tgt = self.pos(tgt)
        src_mask = None
        tgt_mask = self.tgt_mask(Ttgt, src.device)
        enc = src
        for layer in self.encoder:
            enc = layer(enc, src_mask)
        dec = tgt
        for layer in self.decoder:
            dec = layer(dec, enc, src_mask, tgt_mask)
        return self.fc(dec)
# -------------------- Test --------------------
if __name__ == "__main__":
    model = Transformer(1000, 1000)
    src = torch.randint(0, 1000, (2, 10))
    tgt = torch.randint(0, 1000, (2, 10))
    out = model(src, tgt)
    print(out.shape)