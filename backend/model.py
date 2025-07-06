# backend/model.py
import torch
import torch.nn as nn
import numpy as np

class HarmonizerLSTM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # For note probability

    def forward(self, x):
        # x shape: (batch, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        output = self.sigmoid(output)  # Convert to probabilities
        return output

def create_model():
    return HarmonizerLSTM(input_dim=13, hidden_dim=256, output_dim=12)  # Updated dimensions

def load_model(path="models/harmonizer.pt"):
    model = create_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model