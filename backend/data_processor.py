# backend/data_processor.py
import music21
import numpy as np
import torch
from pathlib import Path

class BachDataProcessor:
    def __init__(self):
        self.sequence_length = 32
        self.pitch_classes = 12  # C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        
    def detect_key(self, score):
        """Detect the key of a piece"""
        try:
            key = score.analyze('key')
            return key
        except:
            # Fallback to C major if detection fails
            return music21.key.Key('C', 'major')
    
    def normalize_to_scale_degrees(self, notes, key):
        """Convert absolute pitches to scale degrees"""
        scale_degrees = []
        tonic_pitch_class = key.tonic.pitchClass
        
        for note in notes:
            if note is None:
                scale_degrees.append(-1)  # Rest
            else:
                # Convert to pitch class (0-11)
                pitch_class = note.pitch.pitchClass
                # Calculate scale degree relative to tonic
                scale_degree = (pitch_class - tonic_pitch_class) % 12
                scale_degrees.append(scale_degree)
        
        return scale_degrees
    
    def extract_beat_features(self, score):
        """Extract beat strength features (strong/weak beats)"""
        beats = []
        for element in score.flat.notesAndRests:
            beat_strength = element.beat  # 1.0 = downbeat, 0.5 = weak beat, etc.
            is_strong_beat = 1 if beat_strength == 1.0 else 0
            beats.append(is_strong_beat)
        return beats
    
    def separate_melody_harmony(self, score):
        """Separate melody (soprano) and harmony (alto, tenor, bass)"""
        parts = list(score.parts)
        
        if len(parts) >= 4:  # SATB
            soprano = parts[0]  # Melody
            harmony_parts = parts[1:4]  # Alto, Tenor, Bass
        else:
            # Fallback: treat highest part as melody
            soprano = parts[0] if parts else None
            harmony_parts = parts[1:] if len(parts) > 1 else []
        
        return soprano, harmony_parts
    
    def part_to_sequence(self, part, key, max_length=32):
        """Convert a music21 part to a sequence"""
        notes = []
        beats = []
        
        for element in part.flat.notesAndRests[:max_length]:
            if element.isNote:
                notes.append(element)
                beats.append(1 if element.beat == 1.0 else 0)
            elif element.isChord:
                # Take the highest note of chord for melody
                notes.append(element.notes[-1])
                beats.append(1 if element.beat == 1.0 else 0)
            else:  # Rest
                notes.append(None)
                beats.append(0)
        
        # Pad or truncate to max_length
        while len(notes) < max_length:
            notes.append(None)
            beats.append(0)
        
        # Convert to scale degrees
        scale_degrees = self.normalize_to_scale_degrees(notes[:max_length], key)
        return scale_degrees, beats[:max_length]
    
    def harmony_parts_to_sequence(self, harmony_parts, key, max_length=32):
        """Convert harmony parts to a combined sequence"""
        harmony_matrix = np.zeros((max_length, 12))  # 12 pitch classes
        
        for part in harmony_parts:
            for i, element in enumerate(part.flat.notesAndRests[:max_length]):
                if element.isNote:
                    pitch_class = element.pitch.pitchClass
                    tonic = key.tonic.pitchClass
                    scale_degree = (pitch_class - tonic) % 12
                    harmony_matrix[i, scale_degree] = 1
                elif element.isChord:
                    for note in element.notes:
                        pitch_class = note.pitch.pitchClass
                        tonic = key.tonic.pitchClass
                        scale_degree = (pitch_class - tonic) % 12
                        harmony_matrix[i, scale_degree] = 1
        
        return harmony_matrix
    
    def process_bach_chorales(self, max_pieces=100):
        """Process Bach chorales from music21 corpus"""
        training_data = []
        
        # Get Bach chorales from music21
        bach_chorales = music21.corpus.getComposer('bach')[:max_pieces]
        
        for chorale_path in bach_chorales:
            try:
                print(f"Processing: {chorale_path}")
                score = music21.corpus.parse(chorale_path)
                
                # Detect key
                key = self.detect_key(score)
                
                # Separate melody and harmony
                melody_part, harmony_parts = self.separate_melody_harmony(score)
                
                if melody_part and harmony_parts:
                    # Convert to sequences
                    melody_seq, beat_seq = self.part_to_sequence(melody_part, key)
                    harmony_seq = self.harmony_parts_to_sequence(harmony_parts, key)
                    
                    # Create input features: melody + beat info
                    input_features = np.zeros((self.sequence_length, 13))  # 12 pitch classes + beat
                    
                    for i in range(self.sequence_length):
                        # One-hot encode melody (scale degree)
                        if melody_seq[i] >= 0:  # Not a rest
                            input_features[i, melody_seq[i]] = 1
                        # Add beat information
                        input_features[i, 12] = beat_seq[i]
                    
                    training_data.append({
                        'input': input_features,
                        'target': harmony_seq,
                        'key': str(key)
                    })
                    
            except Exception as e:
                print(f"Error processing {chorale_path}: {e}")
                continue
        
        return training_data