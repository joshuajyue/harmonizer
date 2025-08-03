# backend/model.py
"""
Neural network model architecture for chord harmonization.

This file defines a sophisticated deep learning model that learns to predict
chord progressions from melody inputs. The model uses advanced techniques like
attention mechanisms and bidirectional processing to understand musical context.

Key concepts explained:
- LSTM: A type of neural network good at processing sequences (like melodies)
- Attention: Allows the model to focus on relevant parts of the input
- Residual connections: Help information flow through deep networks
- Layer normalization: Stabilizes training of deep networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SophisticatedChordLSTM(nn.Module):
    """
    Advanced neural network for chord progression prediction.
    
    This model combines several powerful techniques:
    1. LSTM (Long Short-Term Memory): Processes melody sequences step by step
    2. Attention: Lets the model focus on important melody notes when predicting chords
    3. Positional encoding: Helps the model understand beat positions in music
    4. Residual connections: Allow information to skip layers, improving training
    5. Beat-weighted attention: Focuses more on strong beats (chord-defining notes)
    6. Measure-aware processing: Understands musical phrase structure
    
    Architecture flow:
    Melody Input → Project to higher dimension → Add position info → 
    Process with LSTM → Apply beat-weighted attention → Make chord predictions
    
    Parameters:
    - input_dim: Size of melody features (26 features per time step)
    - hidden_dim: Internal processing size (512 for rich representations)
    - output_dim: Number of chord types to predict (7 chord types)
    - num_layers: Depth of LSTM (3 layers for complex patterns)
    - dropout: Regularization to prevent overfitting (30%)
    """
    def __init__(self, input_dim=26, hidden_dim=512, output_dim=7, num_layers=3, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # === INPUT PROCESSING ===
        # Transform melody features from 26 dimensions to 512 for richer representation
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # Normalize inputs for stable training
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # === BEAT IMPORTANCE WEIGHTS ===
        # Learn which beats are most important for chord decisions
        # Strong beats (1, 3) should influence chords more than weak beats (2, 4)
        # Initialize with musical knowledge: beat 1=strongest, beat 3=very strong, 2&4=weak
        initial_beat_weights = torch.tensor([2.5, 1.0, 2.2, 1.0])  # Boost beat 3 more
        self.beat_importance = nn.Parameter(initial_beat_weights)
        
        # === CHORD-DEFINING NOTE DETECTION ===
        # Learn to identify which melody notes define the underlying harmony
        self.chord_defining_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Probability that this note defines the chord
        )
        
        # === SEQUENCE PROCESSING ===
        # Bidirectional LSTM: processes melody both forward and backward in time
        # This helps understand context from both past and future notes
        self.melody_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Combine forward and backward LSTM outputs (2*hidden_dim → hidden_dim)
        self.lstm_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # === MEASURE-AWARE ATTENTION ===
        # Multi-head attention with beat-position awareness
        # Focuses more on chord-defining notes (typically on strong beats)
        self.measure_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.measure_attention_norm = nn.LayerNorm(hidden_dim)
        
        # === CHORD PROGRESSION ATTENTION ===
        # Attention across measures to understand harmonic progressions
        self.progression_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.progression_attention_norm = nn.LayerNorm(hidden_dim)
        
        # === MUSICAL TIMING AWARENESS ===
        # Positional encoding: helps model understand beat positions and timing
        self.position_encoding = PositionalEncoding(hidden_dim, max_len=32)
        
        # === MEASURE GROUPING ===
        # Group time steps into measures for measure-level chord prediction
        self.measure_pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # === HARMONIC CONTEXT ===
        # LSTM for understanding chord progressions across measures
        # Note: Single layer LSTM doesn't use dropout to avoid warnings
        self.harmonic_lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2, 1, 
            batch_first=True  # Remove dropout for single layer
        )
        
        # === CHORD PREDICTION LAYERS ===
        # Multi-layer prediction network focusing on measure-level harmony
        # Combines chord-defining features with harmonic context
        self.chord_fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.chord_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.chord_fc3 = nn.Linear(hidden_dim // 2, output_dim)  # Final layer: 7 chord types
        
        # === REGULARIZATION ===
        # Dropout prevents overfitting by randomly zeroing some neurons during training
        self.dropout = nn.Dropout(dropout)
        
        # Initialize all weights for stable training
        self.init_weights()

    def init_weights(self):
        """
        Initialize neural network weights for stable training.
        
        Uses Xavier initialization which sets weights to reasonable starting values.
        Poor weight initialization can make training slow or unstable.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                # Special initialization for LSTM layers
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        """
        Forward pass: process melody input to predict chord sequence with focus on chord-defining notes.
        
        This version emphasizes:
        1. Strong beats over weak beats (beat 1 and 3 vs 2 and 4)
        2. Chord-defining notes over passing tones
        3. Measure-level harmonic thinking vs note-level
        
        Input: x = melody features, shape (batch_size, sequence_length, 26)
        Output: chord logits, shape (batch_size, sequence_length, 7)
        """
        batch_size, seq_len, _ = x.shape
        
        # === STEP 1: INPUT PROCESSING ===
        # Transform 26-dim melody features to 512-dim rich representation
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = F.relu(x)  # Add non-linearity
        
        # === STEP 2: ADD MUSICAL TIMING INFO ===
        # Inject beat position information into the representation
        x = self.position_encoding(x)
        
        # === STEP 3: DETECT CHORD-DEFINING NOTES ===
        # Identify which melody notes are most important for harmony
        chord_defining_weights = self.chord_defining_detector(x)  # (batch, seq, 1)
        
        # === STEP 4: APPLY BEAT IMPORTANCE ===
        # Weight notes based on their beat position (beat 1 and 3 are stronger)
        beat_positions = torch.arange(seq_len, device=x.device) % 4  # 4 beats per measure
        beat_weights = self.beat_importance[beat_positions]  # (seq_len,)
        beat_weights = beat_weights.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
        # Combine chord-defining and beat importance
        importance_weights = chord_defining_weights * beat_weights
        
        # Apply importance weighting to features
        x_weighted = x * importance_weights
        
        # === STEP 5: SEQUENCE PROCESSING ===
        # Bidirectional LSTM processes weighted melody features
        lstm_out, _ = self.melody_lstm(x_weighted)
        lstm_out = self.lstm_projection(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        # === STEP 6: MEASURE-AWARE ATTENTION ===
        # Focus on important notes within each measure
        measure_attn_out, measure_attn_weights = self.measure_attention(
            lstm_out, lstm_out, lstm_out
        )
        x = self.measure_attention_norm(lstm_out + measure_attn_out)
        
        # === STEP 7: GROUP INTO MEASURES ===
        # Pool every 4 beats (1 measure) for measure-level processing
        # Reshape: (batch, seq_len, hidden) -> (batch, num_measures, beats_per_measure, hidden)
        num_measures = seq_len // 4
        if seq_len % 4 != 0:
            # Pad to complete measures
            padding_needed = 4 - (seq_len % 4)
            x = F.pad(x, (0, 0, 0, padding_needed))
            seq_len = x.shape[1]
            num_measures = seq_len // 4
        
        x_measures = x.view(batch_size, num_measures, 4, self.hidden_dim)
        
        # Pool each measure to single representation (focus on strongest beat)
        measure_features = []
        for i in range(num_measures):
            measure = x_measures[:, i, :, :]  # (batch, 4, hidden)
            # Weight by beat importance and pool
            measure_weights = self.beat_importance.unsqueeze(0).unsqueeze(-1)  # (1, 4, 1)
            weighted_measure = measure * measure_weights
            pooled_measure = torch.mean(weighted_measure, dim=1)  # (batch, hidden)
            measure_features.append(pooled_measure)
        
        measure_sequence = torch.stack(measure_features, dim=1)  # (batch, num_measures, hidden)
        
        # === STEP 8: HARMONIC PROGRESSION PROCESSING ===
        # Understand chord progressions across measures
        harmonic_out, _ = self.harmonic_lstm(measure_sequence)
        
        # === STEP 9: EXPAND BACK TO BEAT LEVEL ===
        # Each measure gets the same chord prediction
        expanded_harmonics = []
        for i in range(num_measures):
            measure_harmony = measure_sequence[:, i:i+1, :]  # (batch, 1, hidden)
            measure_context = harmonic_out[:, i:i+1, :]     # (batch, 1, hidden//2)
            combined_measure = torch.cat([measure_harmony, measure_context], dim=-1)
            # Repeat for each beat in the measure
            expanded_measure = combined_measure.repeat(1, 4, 1)  # (batch, 4, hidden+hidden//2)
            expanded_harmonics.append(expanded_measure)
        
        combined_features = torch.cat(expanded_harmonics, dim=1)  # (batch, seq_len, hidden+hidden//2)
        
        # Trim back to original length if we padded
        if combined_features.shape[1] > seq_len:
            combined_features = combined_features[:, :seq_len, :]
        
        # === STEP 10: CHORD PREDICTION ===
        # Predict chords based on measure-level harmonic understanding
        chord_features = F.relu(self.chord_fc1(combined_features))
        chord_features = self.dropout(chord_features)
        
        chord_features = F.relu(self.chord_fc2(chord_features))
        chord_features = self.dropout(chord_features)
        
        chord_logits = self.chord_fc3(chord_features)
        
        return chord_logits


class PositionalEncoding(nn.Module):
    """
    Positional encoding adds beat/timing information to melody features.
    
    In music, timing is crucial - the same note sounds different on beat 1 vs beat 3.
    This component injects positional information using sine and cosine waves
    of different frequencies, allowing the model to understand musical timing.
    
    Think of it like adding a musical "timestamp" to each note that the AI can understand.
    """
    def __init__(self, d_model, max_len=32):
        super().__init__()
        
        # Create a matrix of positional encodings
        pe = torch.zeros(max_len, d_model)  # (32 time steps, 512 features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create different frequency patterns for each dimension
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        # Use sine for even dimensions, cosine for odd dimensions
        pe[:, 0::2] = torch.sin(position * div_term)  # Even positions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd positions
        
        # Register as buffer (part of model but not trainable)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """Add positional encoding to input features"""
        return x + self.pe[:, :x.size(1)]


def create_model():
    """
    Factory function to create a new model with optimal hyperparameters.
    
    Returns a SophisticatedChordLSTM configured for chord harmonization:
    - 26 input features (melody representation)
    - 512 hidden dimensions (rich internal representation)
    - 7 output classes (different chord types)
    - 3 LSTM layers (sufficient depth for complex patterns)
    - 30% dropout (prevents overfitting)
    """
    return SophisticatedChordLSTM(
        input_dim=26,      # Melody features per time step
        hidden_dim=512,    # Internal processing width
        output_dim=7,      # Number of chord types to predict
        num_layers=3,      # LSTM depth
        dropout=0.3        # Regularization strength
    )


def load_model(model_path):
    """
    Load a pre-trained model from disk.
    
    This function:
    1. Creates a new model with the same architecture
    2. Loads the trained weights from the saved file
    3. Sets the model to evaluation mode (disables training features)
    
    Args:
        model_path: Path to the saved model file (.pt format)
    
    Returns:
        Trained model ready for making predictions
    """
    model = create_model()
    # Load the learned weights onto CPU (works regardless of training device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    # Set to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()
    return model