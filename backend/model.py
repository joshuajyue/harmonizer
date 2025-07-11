# backend/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SophisticatedChordLSTM(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=512, output_dim=7, num_layers=3, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection to higher dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Bidirectional LSTM for better context
        self.melody_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Reduce bidirectional output
        self.lstm_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Multi-head attention with residual connections
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=16, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Position encoding for beat awareness
        self.position_encoding = PositionalEncoding(hidden_dim, max_len=32)
        
        # Multiple attention layers
        self.attention2 = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.attention2_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network with residual connections
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        
        # Context-aware chord prediction
        self.context_lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2, 1, 
            batch_first=True
        )
        
        # Final chord prediction layers with multiple heads
        self.chord_fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.chord_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.chord_fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Input projection and normalization
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = F.relu(x)
        
        # Add positional encoding
        x = self.position_encoding(x)
        
        # Bidirectional LSTM
        lstm_out, _ = self.melody_lstm(x)
        lstm_out = self.lstm_projection(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        # First attention layer with residual connection
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.attention_norm(lstm_out + attn_out)
        
        # Second attention layer with residual connection
        attn_out2, _ = self.attention2(x, x, x)
        x = self.attention2_norm(x + attn_out2)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.ff_norm(x + ff_out)
        
        # Context LSTM for temporal dependencies
        context_out, _ = self.context_lstm(x)
        
        # Concatenate main features with context
        combined = torch.cat([x, context_out], dim=-1)
        
        # Multi-layer chord prediction
        chord_features = F.relu(self.chord_fc1(combined))
        chord_features = self.dropout(chord_features)
        chord_features = F.relu(self.chord_fc2(chord_features))
        chord_features = self.dropout(chord_features)
        chord_logits = self.chord_fc3(chord_features)
        
        return chord_logits  # Return logits, not probabilities


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def create_model():
    """Create a new sophisticated model"""
    return SophisticatedChordLSTM(
        input_dim=26,
        hidden_dim=512,
        output_dim=7,
        num_layers=3,
        dropout=0.3
    )


def load_model(model_path):
    """Load a trained model"""
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model