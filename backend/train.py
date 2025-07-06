# backend/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import HarmonizerLSTM
from data_processor import BachDataProcessor
import os

class BachDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.FloatTensor(item['input']),
            torch.FloatTensor(item['target'])
        )

def train_model():
    # Process data
    print("Processing Bach chorales...")
    processor = BachDataProcessor()
    training_data = processor.process_bach_chorales(max_pieces=50)  # Start small
    
    print(f"Processed {len(training_data)} pieces")
    
    # Create dataset and dataloader
    dataset = BachDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create model
    model = HarmonizerLSTM(input_dim=13, hidden_dim=256, output_dim=12)  # 13 input (12 pitch + beat), 12 output
    criterion = nn.BCELoss()  # Binary cross entropy for multi-label
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/harmonizer_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save(model.state_dict(), 'models/harmonizer.pt')
    print("Training complete!")

if __name__ == "__main__":
    train_model()