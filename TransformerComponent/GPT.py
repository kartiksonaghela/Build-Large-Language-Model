import torch
import torch.nn as nn
import tiktoken

class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, context_length):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_querys = nn.Linear(d_in, d_out, bias=False)
        self.W_keys = nn.Linear(d_in, d_out, bias=False)
        self.W_values = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_projection = nn.Linear(d_out, d_out)
        
        # Causal mask for autoregressive processing
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, inputs):
        batch, num_tokens, dim = inputs.shape
        query = self.W_querys(inputs)
        key = self.W_keys(inputs)
        value = self.W_values(inputs)

        query = query.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask
        mask_bool = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(mask_bool, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = torch.matmul(attn_weights, value)
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch, num_tokens, -1)
        return self.linear_projection(context_vec)

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config['embedding_dim'], config['embedding_dim'] * 4),
            GELU(),
            nn.Linear(config['embedding_dim'] * 4, config['embedding_dim'])
        )

    def forward(self, inputs):
        return self.layers(inputs)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim=True)
        var = inputs.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (inputs - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiheadAttention(
            d_in=config['embedding_dim'], 
            d_out=config['embedding_dim'], 
            dropout=config['dropout'], 
            num_heads=config['n_heads'], 
            context_length=config['context_length']
        )
        self.norm1 = LayerNorm(config["embedding_dim"])
        self.norm2 = LayerNorm(config["embedding_dim"])
        self.ff = FeedForward(config)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, inputs):
        add_connection = inputs
        output = self.norm1(inputs)
        output = self.attention(output)
        output = self.dropout(output)
        output = output + add_connection
        
        add_connection = output
        output = self.norm2(output)
        output = self.ff(output)
        output = self.dropout(output)
        output = output + add_connection
        
        return output

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.pos_embedding = nn.Embedding(config['context_length'], config['embedding_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["n_layers"])])
        self.out_head = nn.Linear(config["embedding_dim"], config["vocab_size"], bias=False)
        self.final_norm = LayerNorm(config["embedding_dim"])

    def forward(self, inputs):
        batch_size, seq_len = inputs.shape
        tok_embeds = self.token_embedding(inputs)
        pos_embeds = self.pos_embedding(torch.arange(seq_len, device=inputs.device))
        
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# GPT Model Configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,    
    "context_length": 1024,  
    "embedding_dim": 768,    
    "n_heads": 12,          
    "n_layers": 12,         
    "dropout": 0.1,         
}

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Tokenizing input text
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch = [torch.tensor(tokenizer.encode(txt1)), torch.tensor(tokenizer.encode(txt2))]
batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)  # Ensure proper batch shape

# Set random seed for reproducibility
torch.manual_seed(123)

# Initialize GPT model
model = GPT(GPT_CONFIG_124M)

# Forward pass
out = model(batch)

# Print results
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")