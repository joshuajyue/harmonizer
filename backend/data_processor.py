# backend/data_processor.py
import music21
import numpy as np
import torch
from pathlib import Path

class MeasureBasedChordProcessor:
    def __init__(self):
        self.sequence_length = 32
        self.chord_types = 7
        self.beats_per_measure = 4
        
        # Chord rhythm patterns (when to change chords within measures)
        self.rhythm_patterns = {
            'simple': [1, 0, 0, 0],      # Change only on downbeat
            'two_chord': [1, 0, 1, 0],   # Change twice per measure
            'active': [1, 0, 1, 1],      # Three changes per measure
            'syncopated': [1, 1, 0, 1],  # Syncopated changes
            'waltz': [1, 0, 0],          # For 3/4 time
        }
        
        # Chord progression scoring weights
        self.progression_scores = {
            'variety_bonus': 2,           # Bonus for using different chords
            'smooth_voice_leading': 3,    # Bonus for smooth transitions
            'functional_harmony': 4,      # Bonus for traditional progressions
            'dissonance_penalty': -2,     # Penalty for harsh transitions
            'repetition_penalty': -1,     # Penalty for too much repetition
        }
    
    def analyze_melody_rhythm_pattern(self, melody_notes, key):
        """Analyze melody to determine optimal chord change rhythm"""
        if not melody_notes:
            return 'simple'
        
        # Count note density and rhythmic activity
        note_density = len([n for n in melody_notes if n is not None]) / len(melody_notes)
        
        # Analyze pitch movement (melodic activity)
        pitch_changes = 0
        prev_pitch = None
        for note in melody_notes:
            if note is not None and prev_pitch is not None:
                if abs(note.pitch.midi - prev_pitch) > 2:  # Significant pitch change
                    pitch_changes += 1
                prev_pitch = note.pitch.midi
            elif note is not None:
                prev_pitch = note.pitch.midi
        
        activity_ratio = pitch_changes / max(1, len(melody_notes))
        
        # Determine rhythm pattern based on activity
        if note_density > 0.8 and activity_ratio > 0.4:
            return 'active'
        elif note_density > 0.6 and activity_ratio > 0.3:
            return 'two_chord'
        elif activity_ratio > 0.5:
            return 'syncopated'
        else:
            return 'simple'
    
    def generate_creative_chord_progression(self, melody_measures, key, num_measures):
        """Generate progression for ACTUAL number of measures - one chord per measure"""
        is_minor = key.mode == 'minor'
        progression = []
        
        print(f"Generating progression for {num_measures} ACTUAL measures")
        
        # Generate ONE chord per measure, then expand later
        measure_chords = []
        for measure_idx in range(num_measures):
            melody_pitch_classes = melody_measures[measure_idx]
            
            # Find best chord for this measure
            prev_chord = measure_chords[-1] if measure_chords else None
            best_chord = self.find_chord_for_melody_context_pitches(
                melody_pitch_classes, key, prev_chord
            )
            measure_chords.append(best_chord)
            print(f"Measure {measure_idx + 1}: Chord {best_chord}")
        
        print(f"Generated {len(measure_chords)} chords for {num_measures} measures: {measure_chords}")
        return measure_chords  # Return just the measure-level chords

    def find_chord_for_melody_context_pitches(self, melody_pitch_classes, key, prev_chord):
        """Find the best chord based on pitch class integers (not Note objects)"""
        print(f"    Analyzing pitch classes: {melody_pitch_classes}")
        
        if not melody_pitch_classes:
            print(f"    No melody notes found, defaulting to tonic (0)")
            return 0  # Default to tonic
        
        tonic = key.tonic.pitchClass
        is_minor = key.mode == 'minor'
        
        # Convert pitch classes to scale degrees
        scale_degrees = []
        for pc in melody_pitch_classes:
            scale_degree = (pc - tonic) % 12
            scale_degrees.append(scale_degree)
        
        if not scale_degrees:
            print(f"    No valid pitches found, defaulting to tonic (0)")
            return 0
        
        print(f"    Scale degrees: {scale_degrees}")
        
        # Score each possible chord based on how well it fits the melody
        chord_scores = {}
        
        for chord_type in range(7):
            chord_tones = self.get_chord_tones(chord_type, is_minor)
            score = 0
            
            # Count how many melody notes are chord tones
            for scale_degree in scale_degrees:
                if scale_degree in chord_tones:
                    score += 3  # Strong fit
                elif self.is_neighbor_tone(scale_degree, chord_tones):
                    score += 1  # Passing tone/neighbor
                else:
                    score -= 1  # Dissonance
            
            # Bonus for functional harmony progression
            if prev_chord is not None:
                if self.is_functional_progression(prev_chord, chord_type, is_minor):
                    score += 2
            
            # Slight preference for more common chords in first measure
            if prev_chord is None:  # First chord
                if chord_type in [0, 3, 4]:  # I, IV, V (or i, iv, V)
                    score += 1
            
            chord_scores[chord_type] = score
            print(f"      Chord {chord_type}: tones={chord_tones}, score={score}")
        
        # Select best chord
        best_chord = max(chord_scores, key=chord_scores.get)
        best_score = chord_scores[best_chord]
        
        print(f"    Selected chord {best_chord} with score {best_score}")
        return best_chord
    
    def is_neighbor_tone(self, pitch, chord_tones):
        """Check if a pitch is a neighbor tone to any chord tone"""
        for chord_tone in chord_tones:
            if abs(pitch - chord_tone) == 1 or abs(pitch - chord_tone) == 11:
                return True
        return False
    
    def extract_measure_based_features(self, melody_part, key):
        """Extract features with proper timing resolution"""
        
        # Get the time signature and calculate proper beat divisions
        time_sig = melody_part.getTimeSignatures()[0] if melody_part.getTimeSignatures() else music21.meter.TimeSignature('4/4')
        beats_per_measure = time_sig.numerator
        
        # Calculate total length in QUARTER NOTES (not every note event)
        total_quarters = int(melody_part.duration.quarterLength)
        print(f"Total piece length: {total_quarters} quarter notes")
        
        # Create beat-level features (one feature vector per quarter note)
        features = []
        chord_progression = []
        
        # Process in quarter-note increments
        for quarter_beat in range(total_quarters):
            # Get all notes that are sounding at this quarter-note position
            beat_start = quarter_beat
            beat_end = quarter_beat + 1
            
            # Find notes active during this quarter note
            active_notes = []
            for note in melody_part.flat.notes:
                note_start = note.offset
                note_end = note.offset + note.duration.quarterLength
                
                # Check if note overlaps with this quarter-note beat
                if note_start < beat_end and note_end > beat_start:
                    if hasattr(note, 'pitch'):  # Single note
                        active_notes.append(note.pitch.pitchClass)
                    elif hasattr(note, 'pitches'):  # Chord
                        active_notes.extend([p.pitchClass for p in note.pitches])
        
            # Create feature vector for this quarter note
            pitch_class_vector = np.zeros(12)
            for pc in active_notes:
                pitch_class_vector[pc] = 1
            
            # Add timing and key features
            beat_in_measure = quarter_beat % beats_per_measure
            strong_beat = 1 if beat_in_measure == 0 else 0
            
            # Key encoding
            tonic_pc = key.tonic.pitchClass if key else 0
            is_minor = 1 if (key and key.mode == 'minor') else 0
            
            # Combine all features
            beat_features = np.concatenate([
                pitch_class_vector,     # 12 dims: pitch classes
                [strong_beat],          # 1 dim: strong beat indicator
                [is_minor],             # 1 dim: major/minor
                np.zeros(12)            # 12 dims: reserved for additional features
            ])
            
            features.append(beat_features)
        
        # Convert to numpy array
        features = np.array(features)
        actual_length = len(features)
        
        print(f"Created {actual_length} quarter-note beats from {total_quarters} quarter notes")
        
        # Generate chord progression at quarter-note level (much more reasonable!)
        if actual_length > 0:
            # Group quarter notes into measures for chord analysis
            measures = []
            notes_per_measure = beats_per_measure
            
            for measure_start in range(0, actual_length, notes_per_measure):
                measure_end = min(measure_start + notes_per_measure, actual_length)
                measure_pitch_classes = []
                
                for beat in range(measure_start, measure_end):
                    active_pitches = [i for i, val in enumerate(features[beat][:12]) if val > 0]
                    measure_pitch_classes.extend(active_pitches)
                
                # Remove duplicates and store as list of integers
                unique_pitches = list(set(measure_pitch_classes))
                measures.append(unique_pitches)
                print(f"Measure {len(measures)}: pitch classes = {unique_pitches}")
            
            # Generate chord progression using the fixed function
            measure_chords = self.generate_creative_chord_progression(measures, key, len(measures))
            
            # Expand to quarter-note level - each measure chord lasts for notes_per_measure beats
            chord_progression = []
            for measure_idx in range(len(measures)):
                measure_chord = measure_chords[measure_idx] if measure_idx < len(measure_chords) else 0
                
                # This measure's chord applies to all beats in the measure
                for beat_in_measure in range(notes_per_measure):
                    if len(chord_progression) < actual_length:
                        # Add some variation: change chord on beat 3 of some measures
                        if beat_in_measure == 2 and measure_idx % 2 == 1:  # Beat 3 of odd measures
                            # Use a related chord (dominant or subdominant)
                            if measure_chord == 0:  # If tonic, go to dominant
                                varied_chord = 4
                            elif measure_chord == 4:  # If dominant, go to tonic
                                varied_chord = 0  
                            else:  # Otherwise use tonic
                                varied_chord = 0
                            chord_progression.append(varied_chord)
                        else:
                            chord_progression.append(measure_chord)

            print(f"Expanded to beat level: {chord_progression[:16]}... (length: {len(chord_progression)})")
        else:
            chord_progression = [0] * actual_length
        
        print(f"Generated chord progression: {chord_progression[:16]}... (length: {len(chord_progression)})")
        
        # Convert chord progression to one-hot
        chord_labels = np.zeros((actual_length, 7))
        for i, chord in enumerate(chord_progression):
            if i < actual_length:
                chord_labels[i, chord] = 1
        
        # Pad to fixed size for training
        target_length = 32
        if actual_length < target_length:
            feature_padding = np.zeros((target_length - actual_length, features.shape[1]))
            chord_padding = np.zeros((target_length - actual_length, 7))
            chord_padding[:, 0] = 1  # Default to tonic
            
            features = np.vstack([features, feature_padding])
            chord_labels = np.vstack([chord_labels, chord_padding])
        else:
            features = features[:target_length]
            chord_labels = chord_labels[:target_length]
        
        return features, chord_labels, actual_length
    
    def process_bach_chorales(self, max_pieces=100):
        """Process Bach chorales for training data with better error handling"""
        training_data = []
        processed_count = 0
        target_length = 32  # Fixed sequence length for training
        
        print("Getting Bach chorales...")
        try:
            chorale_list = music21.corpus.getComposer('bach')[:max_pieces]
        except Exception as e:
            print(f"Error getting Bach corpus: {e}")
            return []
        
        for chorale_path in chorale_list:
            if processed_count >= max_pieces:
                break
                
            try:
                print(f"Processing: {chorale_path}")
                score = music21.corpus.parse(chorale_path)
                
                # Extract melody (soprano)
                parts = list(score.parts)
                if len(parts) == 0:
                    continue
                    
                melody_part = parts[0]  # Soprano
                
                # Detect key
                key = self.detect_key(score)
                if key is None:
                    print(f"Could not detect key for {chorale_path}")
                    continue
                
                # Extract features - HANDLE THE 3 RETURN VALUES
                try:
                    features, chord_labels, actual_length = self.extract_measure_based_features(melody_part, key)
                    
                    # Skip pieces that are too short
                    if actual_length < 8:
                        continue
                    
                    # Pad or truncate to target_length
                    if actual_length > target_length:
                        # Truncate
                        features = features[:target_length]
                        chord_labels = chord_labels[:target_length]
                    elif actual_length < target_length:
                        # Pad with zeros
                        feature_padding = np.zeros((target_length - actual_length, features.shape[1]))
                        chord_padding = np.zeros((target_length - actual_length, chord_labels.shape[1]))
                        # For chord padding, use tonic chord (index 0)
                        chord_padding[:, 0] = 1
                        
                        features = np.vstack([features, feature_padding])
                        chord_labels = np.vstack([chord_labels, chord_padding])
                    
                    # Verify final shape
                    assert features.shape == (target_length, 26), f"Wrong feature shape: {features.shape}"
                    assert chord_labels.shape == (target_length, 7), f"Wrong label shape: {chord_labels.shape}"
                    
                    training_data.append({
                        'input': features,
                        'target': chord_labels
                    })
                    processed_count += 1
                    print(f"Successfully processed piece {processed_count}: padded to {target_length} beats")
                    
                except Exception as e:
                    print(f"Error extracting features from {chorale_path}: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error processing {chorale_path}: {e}")
                continue
        
        print(f"Successfully processed {len(training_data)} chorales")
        return training_data
    
    def get_chord_tones(self, chord_func, is_minor=False):
        """Get the chord tones for a given chord function"""
        if is_minor:
            chord_patterns = {
                0: [0, 3, 7],    # i (minor tonic)
                1: [2, 5, 8],    # ii° (diminished)
                2: [3, 7, 10],   # III (major)
                3: [5, 8, 0],    # iv (minor)
                4: [7, 11, 2],   # V (major - dominant)
                5: [8, 0, 3],    # VI (major)
                6: [10, 2, 5],   # VII (major)
            }
        else:
            chord_patterns = {
                0: [0, 4, 7],    # I
                1: [2, 5, 9],    # ii
                2: [4, 7, 11],   # iii
                3: [5, 9, 0],    # IV
                4: [7, 11, 2],   # V
                5: [9, 0, 4],    # vi
                6: [11, 2, 5],   # vii°
            }
        
        return chord_patterns.get(chord_func, [0, 4, 7])
    
    def encode_key(self, key):
        """Encode key as one-hot vector"""
        key_encoding = np.zeros(12)
        tonic_pitch_class = key.tonic.pitchClass
        key_encoding[tonic_pitch_class] = 1
        return key_encoding

    def detect_key(self, score):
        """Detect the key of a piece"""
        try:
            key = score.analyze('key')
            return key
        except:
            return music21.key.Key('C', 'major')

    def separate_melody_harmony(self, score):
        """Separate melody and harmony parts"""
        parts = list(score.parts)
        if len(parts) >= 4:  # SATB
            soprano = parts[0]
            harmony_parts = parts[1:4]
        else:
            soprano = parts[0] if parts else None
            harmony_parts = parts[1:] if len(parts) > 1 else []
        return soprano, harmony_parts
    
    def is_functional_progression(self, chord1, chord2, is_minor):
        """Check if chord progression follows functional harmony"""
        if is_minor:
            # Common progressions in minor keys
            functional_progressions = {
                (0, 3): True,  # i -> iv
                (0, 4): True,  # i -> V
                (0, 5): True,  # i -> VI
                (3, 0): True,  # iv -> i
                (3, 4): True,  # iv -> V
                (4, 0): True,  # V -> i
                (4, 5): True,  # V -> VI
                (5, 3): True,  # VI -> iv
                (5, 4): True,  # VI -> V
                (1, 4): True,  # ii° -> V
                (2, 5): True,  # III -> VI
                (6, 0): True,  # VII -> i
                (6, 4): True,  # VII -> V
            }
        else:
            # Common progressions in major keys
            functional_progressions = {
                (0, 4): True,  # I -> V
                (0, 3): True,  # I -> IV  
                (0, 5): True,  # I -> vi
                (0, 1): True,  # I -> ii
                (4, 0): True,  # V -> I
                (3, 0): True,  # IV -> I
                (3, 4): True,  # IV -> V
                (5, 3): True,  # vi -> IV
                (5, 4): True,  # vi -> V
                (1, 4): True,  # ii -> V
                (1, 0): True,  # ii -> I
                (2, 5): True,  # iii -> vi
                (2, 1): True,  # iii -> ii
                (6, 0): True,  # vii° -> I
                (6, 4): True,  # vii° -> V
            }
        
        return functional_progressions.get((chord1, chord2), False)

    def get_functional_next_chords(self, current_chord, is_minor):
        """Get functionally appropriate next chords"""
        if is_minor:
            next_chords_map = {
                0: [3, 4, 5],    # i -> iv, V, VI
                1: [4, 0],       # ii° -> V, i
                2: [5, 6],       # III -> VI, VII
                3: [4, 0],       # iv -> V, i
                4: [0, 5],       # V -> i, VI
                5: [3, 4],       # VI -> iv, V
                6: [0, 4],       # VII -> i, V
            }
        else:
            next_chords_map = {
                0: [3, 4, 5, 1], # I -> IV, V, vi, ii
                1: [4, 0],       # ii -> V, I
                2: [5, 1],       # iii -> vi, ii
                3: [4, 0, 1],    # IV -> V, I, ii
                4: [0, 5],       # V -> I, vi
                5: [3, 1, 4],    # vi -> IV, ii, V
                6: [0, 4],       # vii° -> I, V
            }
        
        return next_chords_map.get(current_chord, [0, 4])

    def calculate_voice_leading_score(self, chord1, chord2, key):
        """Calculate voice leading smoothness between two chords"""
        is_minor = key.mode == 'minor'
        
        tones1 = self.get_chord_tones(chord1, is_minor)
        tones2 = self.get_chord_tones(chord2, is_minor)
        
        # Calculate minimum voice movement
        total_movement = 0
        for tone1 in tones1:
            min_movement = min(abs(tone1 - tone2) % 12 for tone2 in tones2)
            total_movement += min(min_movement, 12 - min_movement)  # Consider both directions
        
        # Lower movement = higher score
        return self.progression_scores['smooth_voice_leading'] * (7 - total_movement)