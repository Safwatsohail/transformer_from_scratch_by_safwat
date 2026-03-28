import torch 
import torch.nn as nn 
import math 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead 
        self.d_k  = d_model // nhead
        
        # Linear projections for query, key, and value vectors
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 1. Linear map & reshape to (batch_size, nhead, seq_len, d_k)
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)    

        # 2. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None: 
            scores = scores.masked_fill(mask == 0, float("-1e20"))
                
        attention = torch.softmax(scores, dim=-1)

        # Apply dropout to attention weights and multiply by values
        x = torch.matmul(self.dropout(attention), V)

        # 3. Concatenate heads and pass through final linear projection
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_out(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x): 
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x 


class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(EncoderBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Apply self-attention, residual connection, and layer normalization
        original_x = x
        attention_output = self.attention(x, x, x, mask)
        x = self.norm_1(original_x + self.dropout(attention_output))

        # Apply feed forward, residual connection, and layer normalization
        original_x = x
        ff_output = self.feed_forward(x)
        x = self.norm_2(original_x + self.dropout(ff_output))
        
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate frequencies for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x): 
        x = x + self.pe[:, :x.size(1)]
        return x 


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, d_ff, max_seq_length, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(d_model, nhead, d_ff, dropout) for _ in range(num_encoder_layers)]
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        src = self.embeddings(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
            
        output = self.fc_out(src)
        return output 


if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 1000
    d_model = 512
    nhead = 8
    num_layers = 6
    d_ff = 2048
    max_len = 100
    
    # Initialize model
    model = TransformerModel(vocab_size, d_model, nhead, num_layers, d_ff, max_len)
    
    # Generate dummy input tensor of shape (batch_size, seq_len)
    dummy_input = torch.randint(0, vocab_size, (2, 10))
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}") 
    print(f"Output shape: {output.shape}") 
