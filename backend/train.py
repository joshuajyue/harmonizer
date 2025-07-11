# backend/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import SophisticatedChordLSTM, create_model
from data_processor import MeasureBasedChordProcessor
import os

class BachDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.FloatTensor(item['input']),    # Shape: (32, 26)
            torch.FloatTensor(item['target'])    # Shape: (32, 7)
        )

def train_chord_model():
    # Process data
    print("Processing Bach chorales...")
    processor = MeasureBasedChordProcessor()
    training_data = processor.process_bach_chorales(max_pieces=100)  # More data
    
    print(f"Processed {len(training_data)} pieces")
    
    if len(training_data) == 0:
        print("No training data found!")
        return
    
    # Create dataset and dataloader
    dataset = BachDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Larger batch size
    
    # Create sophisticated model
    model = create_model()
    
    # Better loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Better optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    print("Starting training with sophisticated model...")
    
    # Training loop
    num_epochs = 200  # More epochs for sophisticated model
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass - model now returns logits
            outputs = model(inputs)  # Shape: (batch, 32, 7)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, 7)          # (batch*32, 7)
            targets = targets.view(-1, 7)          # (batch*32, 7)
            target_indices = torch.argmax(targets, dim=1)  # (batch*32,)
            
            # Calculate loss
            loss = criterion(outputs, target_indices)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/chord_harmonizer_best.pt')
            print(f"New best model saved with loss: {best_loss:.4f}")
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/chord_harmonizer_epoch_{epoch+1}.pt')
            print(f"Saved checkpoint at epoch {epoch+1}")
            
            # Evaluate model diversity on a sample
            model.eval()
            with torch.no_grad():
                sample_input, sample_target = dataset[0]
                sample_output = model(sample_input.unsqueeze(0))
                sample_probs = torch.softmax(sample_output.squeeze(0), dim=-1)
                sample_predictions = torch.argmax(sample_probs, dim=-1).numpy()
                
                print(f"Sample predictions: {sample_predictions[:16]}")
                print(f"Prediction diversity: {len(set(sample_predictions))} unique chords out of 7")
            model.train()
        
        # Early stopping if loss stops improving
        if epoch > 50 and avg_loss > best_loss * 1.1:
            print("Loss stopped improving, implementing early stopping...")
            break
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/chord_harmonizer.pt')
    print("Training complete! Model saved as 'models/chord_harmonizer.pt'")
    
    # Final evaluation
    print("\nFinal model evaluation:")
    model.eval()
    with torch.no_grad():
        total_accuracy = 0
        total_diversity = 0
        
        for i in range(min(10, len(dataset))):
            sample_input, sample_target = dataset[i]
            sample_output = model(sample_input.unsqueeze(0))
            sample_probs = torch.softmax(sample_output.squeeze(0), dim=-1)
            sample_predictions = torch.argmax(sample_probs, dim=-1).numpy()
            target_predictions = torch.argmax(sample_target, dim=-1).numpy()
            
            # Calculate accuracy
            accuracy = np.mean(sample_predictions == target_predictions)
            total_accuracy += accuracy
            
            # Calculate diversity
            diversity = len(set(sample_predictions)) / 7.0
            total_diversity += diversity
        
        avg_accuracy = total_accuracy / min(10, len(dataset))
        avg_diversity = total_diversity / min(10, len(dataset))
        
        print(f"Average accuracy: {avg_accuracy:.3f}")
        print(f"Average diversity: {avg_diversity:.3f}")


if __name__ == "__main__":
    train_chord_model()