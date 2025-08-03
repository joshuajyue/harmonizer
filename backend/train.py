# backend/train.py
"""
Training script for the chord harmonization neural network.

This script trains an AI model to predict chord progressions based on melody inputs,
using Bach chorales as training data. The model learns to harmonize melodies
in the style of Johann Sebastian Bach.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import SophisticatedChordLSTM, create_model
from data_processor import MeasureBasedChordProcessor
import os

class BachDataset(Dataset):
    """
    Custom dataset class for Bach chorale data.
    
    This wraps our processed Bach chorale data so PyTorch can efficiently
    load it in batches during training. Each item contains:
    - input: melody information (32 time steps, 26 features each)
    - target: corresponding chord labels (32 time steps, 7 chord types)
    """
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.FloatTensor(item['input']),    # Shape: (32, 26) - melody features
            torch.FloatTensor(item['target'])    # Shape: (32, 7) - chord labels
        )

def train_chord_model():
    """
    Main training function for the chord harmonization model.
    
    This function:
    1. Loads and processes Bach chorale training data
    2. Creates and configures the neural network model
    3. Trains the model to predict chord progressions from melodies
    4. Saves the best performing model for later use
    """
    # === DATA PREPARATION ===
    # Load Bach chorales and convert them to training examples
    print("Processing Bach chorales...")
    processor = MeasureBasedChordProcessor()
    training_data = processor.process_bach_chorales(max_pieces=100)  # Use 100 pieces for training
    
    print(f"Processed {len(training_data)} pieces")
    
    if len(training_data) == 0:
        print("No training data found!")
        return
    
    # === MODEL SETUP ===
    # Create dataset wrapper and data loader for efficient batch processing
    dataset = BachDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Process 16 examples at once
    
    # Create the neural network model
    model = create_model()
    
    # === TRAINING CONFIGURATION ===
    # Analyze class distribution to handle imbalance
    print("Analyzing class distribution...")
    all_targets = []
    for data in training_data:
        targets = torch.argmax(torch.FloatTensor(data['target']), dim=-1)
        all_targets.extend(targets.numpy())
    
    # Calculate class weights to handle imbalance
    from collections import Counter
    class_counts = Counter(all_targets)
    print(f"Class distribution: {class_counts}")
    
    # Create inverse frequency weights (but cap them to avoid extreme values)
    total_samples = len(all_targets)
    class_weights = torch.zeros(7)
    for i in range(7):
        count = class_counts.get(i, 1)  # Avoid division by zero
        raw_weight = total_samples / (7 * count)  # Inverse frequency weighting
        # Cap weights to prevent extreme overcompensation (max 5x normal weight)
        class_weights[i] = min(raw_weight, 5.0)
    
    print(f"Class weights (capped): {class_weights}")
    
    # Loss function with class weights and label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Optimizer: algorithm that adjusts model weights to reduce loss
    # AdamW includes weight decay to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)  # Lower LR for stability
    
    # Learning rate scheduler: automatically reduces learning rate when training plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=8
    )
    
    print("Starting training with sophisticated model...")
    
    # === TRAINING LOOP ===
    # Train for up to 200 epochs (complete passes through the data)
    num_epochs = 200
    best_loss = float('inf')  # Track the best performance so far
    
    for epoch in range(num_epochs):
        # Set model to training mode (enables dropout, batch norm, etc.)
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Process all training data in batches
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Reset gradients from previous batch
            optimizer.zero_grad()
            
            # === FORWARD PASS ===
            # Feed melody data through the model to get chord predictions
            outputs = model(inputs)  # Shape: (batch_size, 32, 7) - 7 possible chord types
            
            # Reshape data for loss calculation
            outputs = outputs.view(-1, 7)          # Flatten to (batch*32, 7)
            targets = targets.view(-1, 7)          # Flatten to (batch*32, 7)
            target_indices = torch.argmax(targets, dim=1)  # Convert to class indices
            
            # === LOSS CALCULATION ===
            # Calculate primary prediction loss with class balancing
            prediction_loss = criterion(outputs, target_indices)
            
            # === CHORD PROGRESSION SMOOTHNESS LOSS ===
            # Encourage reasonable chord progressions (discourage random jumping)
            current_batch_size = outputs.shape[0] // 32  # outputs are flattened, so infer batch size
            batch_chord_predictions = torch.argmax(outputs.view(current_batch_size, -1, 7), dim=-1)  # (batch, seq)
            
            progression_loss = 0
            if batch_chord_predictions.shape[1] > 1:
                # Penalize too many chord changes (encourage measure-level thinking)
                chord_changes = (batch_chord_predictions[:, 1:] != batch_chord_predictions[:, :-1]).float()
                
                # Calculate overall change rate
                change_rate = torch.mean(chord_changes)
                base_progression_loss = torch.abs(change_rate - 0.30) * 0.5  # Increased target to 30%
                
                # BONUS: Specifically encourage changes on beat 3 (positions 2, 6, 10, 14, 18, 22, 26, 30)
                seq_len = batch_chord_predictions.shape[1]
                beat_3_positions = torch.tensor([i for i in range(2, seq_len, 4) if i < seq_len - 1])  # Beat 3 positions
                
                if len(beat_3_positions) > 0:
                    beat_3_changes = chord_changes[:, beat_3_positions]  # Changes that happen ON beat 3
                    beat_3_change_rate = torch.mean(beat_3_changes)
                    # Encourage MORE changes on beat 3 (target 60% change rate on beat 3)
                    beat_3_bonus = torch.abs(beat_3_change_rate - 0.60) * 0.3
                    progression_loss = base_progression_loss + beat_3_bonus
                else:
                    progression_loss = base_progression_loss
            
            # === DIVERSITY ENCOURAGEMENT ===
            # Prevent collapse to single chord by encouraging diversity within sequences
            probs = torch.softmax(outputs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            diversity_loss = -torch.mean(entropy) * 0.05  # Small weight to encourage diversity
            
            # === CHORD TYPE BALANCE LOSS ===
            # Prevent overuse of specific chord types (like V and vii°)
            batch_predictions = torch.argmax(outputs, dim=-1)
            chord_distribution = torch.bincount(batch_predictions, minlength=7).float()
            chord_distribution = chord_distribution / torch.sum(chord_distribution)
            
            # Target a more balanced distribution (not perfectly uniform, but not extreme)
            # Encourage I, IV, V as common, ii, iii, vi as medium, vii° as rare
            target_distribution = torch.tensor([0.25, 0.15, 0.10, 0.20, 0.20, 0.08, 0.02])  # Musically reasonable
            balance_loss = torch.mean((chord_distribution - target_distribution) ** 2) * 0.1
            
            # Combine all losses
            total_batch_loss = prediction_loss + progression_loss + diversity_loss + balance_loss
            
            # === BACKWARD PASS ===
            # Calculate gradients and update model weights
            total_batch_loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
            
            # Print progress periodically
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], '
                      f'Total: {total_batch_loss.item():.4f} '
                      f'(Pred: {prediction_loss.item():.4f}, '
                      f'Prog: {progression_loss:.4f}, '
                      f'Div: {diversity_loss.item():.4f}, '
                      f'Bal: {balance_loss.item():.4f})')
                
                # Show current chord distribution in batch
                pred_chords = torch.argmax(outputs, dim=-1)
                unique_chords, counts = torch.unique(pred_chords, return_counts=True)
                chord_percentages = (counts.float() / len(pred_chords) * 100)
                chord_info = [f"{chord.item()}:{pct:.1f}%" for chord, pct in zip(unique_chords, chord_percentages)]
                print(f'    Chord distribution: {", ".join(chord_info)}')
                
                # Show beat 3 change rate for monitoring
                if batch_chord_predictions.shape[1] > 1:
                    beat_3_positions = torch.tensor([i for i in range(2, batch_chord_predictions.shape[1], 4) if i < batch_chord_predictions.shape[1] - 1])
                    if len(beat_3_positions) > 0:
                        beat_3_changes = chord_changes[:, beat_3_positions]
                        beat_3_rate = torch.mean(beat_3_changes).item()
                        print(f'    Beat 3 change rate: {beat_3_rate:.2f} (target 0.60)')
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        # Update learning rate based on performance
        scheduler.step(avg_loss)
        
        # === MODEL CHECKPOINTING ===
        # Save the best performing model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/chord_harmonizer_best.pt')
            print(f"New best model saved with loss: {best_loss:.4f}")
        
        # Save periodic checkpoints every 25 epochs
        if (epoch + 1) % 25 == 0:
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/chord_harmonizer_epoch_{epoch+1}.pt')
            print(f"Saved checkpoint at epoch {epoch+1}")
            
            # === MODEL EVALUATION ===
            # Test the model's diversity and progression quality
            model.eval()  # Switch to evaluation mode
            with torch.no_grad():  # Don't calculate gradients during evaluation
                sample_input, sample_target = dataset[0]
                sample_output = model(sample_input.unsqueeze(0))
                sample_probs = torch.softmax(sample_output.squeeze(0), dim=-1)
                sample_predictions = torch.argmax(sample_probs, dim=-1).numpy()
                
                print(f"Sample predictions: {sample_predictions[:16]}")
                print(f"Prediction diversity: {len(set(sample_predictions))} unique chords out of 7")
                
                # Analyze chord progression patterns
                chord_changes = sum(1 for i in range(1, len(sample_predictions)) 
                                  if sample_predictions[i] != sample_predictions[i-1])
                change_rate = chord_changes / (len(sample_predictions) - 1)
                print(f"Chord change rate: {change_rate:.3f} (target ~0.30)")
                
                # Analyze beat 3 change patterns specifically
                beat_3_changes = sum(1 for i in range(2, len(sample_predictions), 4) 
                                   if i < len(sample_predictions) and i > 0 and 
                                   sample_predictions[i] != sample_predictions[i-1])
                total_beat_3_opportunities = len([i for i in range(2, len(sample_predictions), 4)])
                if total_beat_3_opportunities > 0:
                    beat_3_change_rate = beat_3_changes / total_beat_3_opportunities
                    print(f"Beat 3 change rate: {beat_3_change_rate:.3f} (target ~0.60)")
                
                # Check if model is stuck on single chord
                most_common_chord = max(set(sample_predictions), key=list(sample_predictions).count)
                most_common_count = list(sample_predictions).count(most_common_chord)
                dominance_ratio = most_common_count / len(sample_predictions)
                print(f"Most common chord {most_common_chord}: {dominance_ratio:.3f} dominance")
                
                if dominance_ratio > 0.8:
                    print("WARNING: Model may be overfitting to single chord!")
                
            model.train()  # Switch back to training mode
        
        # === EARLY STOPPING ===
        # Stop training if the model stops improving (prevents overfitting)
        if epoch > 50 and avg_loss > best_loss * 1.1:
            print("Loss stopped improving, implementing early stopping...")
            break
    
    # === FINAL MODEL SAVING ===
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/chord_harmonizer.pt')
    print("Training complete! Model saved as 'models/chord_harmonizer.pt'")
    
    # === COMPREHENSIVE FINAL EVALUATION ===
    print("\nFinal model evaluation:")
    model.eval()
    with torch.no_grad():
        total_accuracy = 0
        total_diversity = 0
        
        # Test on multiple samples to get robust metrics
        num_test_samples = min(10, len(dataset))
        for i in range(num_test_samples):
            sample_input, sample_target = dataset[i]
            sample_output = model(sample_input.unsqueeze(0))
            sample_probs = torch.softmax(sample_output.squeeze(0), dim=-1)
            sample_predictions = torch.argmax(sample_probs, dim=-1).numpy()
            target_predictions = torch.argmax(sample_target, dim=-1).numpy()
            
            # Calculate accuracy: how often our predictions match the target
            accuracy = np.mean(sample_predictions == target_predictions)
            total_accuracy += accuracy
            
            # Calculate diversity: how many different chords we predict
            diversity = len(set(sample_predictions)) / 7.0  # Normalize by max possible (7 chord types)
            total_diversity += diversity
        
        # Report average performance metrics
        avg_accuracy = total_accuracy / num_test_samples
        avg_diversity = total_diversity / num_test_samples
        
        print(f"Average accuracy: {avg_accuracy:.3f} (how often we match Bach exactly)")
        print(f"Average diversity: {avg_diversity:.3f} (variety in chord usage, 0-1 scale)")
        
        # === ADVANCED DIVERSITY ANALYSIS ===
        # Analyze the model's final predictions for musical quality
        sample_input, _ = dataset[0]
        sample_output = model(sample_input.unsqueeze(0))
        
        # Calculate entropy (randomness/diversity) of predictions
        probs = torch.softmax(sample_output, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        diversity_score = torch.mean(entropy)
        
        # Analyze chord transitions (musical smoothness)
        if sample_output.shape[1] > 1:  # If we have multiple time steps
            pred_chords = torch.argmax(sample_output, dim=-1)  # Get predicted chord sequence
            # Count how often chords change (vs. staying the same)
            transitions = pred_chords[:, 1:] != pred_chords[:, :-1]  
            transition_rate = torch.mean(transitions.float())
        else:
            transition_rate = 0
        
        print(f"Entropy score: {diversity_score:.4f} (higher = more diverse)")
        print(f"Chord transition rate: {transition_rate:.4f} (how often chords change)")
        print("\nTraining completed successfully!")


if __name__ == "__main__":
    train_chord_model()