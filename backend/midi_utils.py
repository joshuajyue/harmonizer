# backend/midi_utils.py
import mido
import numpy as np
import music21

def detect_key_from_midi(midi_file):
    """Detect the key of a MIDI file"""
    try:
        # Convert MIDI to music21 score for key analysis
        score = music21.converter.parse(midi_file)
        key = score.analyze('key')
        return key
    except:
        # Fallback to C major
        return music21.key.Key('C', 'major')

def encode_key(key):
    """Encode key as one-hot vector (12 dimensions for 12 keys)"""
    key_encoding = np.zeros(12)
    tonic_pitch_class = key.tonic.pitchClass
    key_encoding[tonic_pitch_class] = 1
    return key_encoding

def chord_to_midi_notes(chord_type, key):
    """Convert chord type to MIDI notes based on key (supports major and minor)"""
    tonic = key.tonic.pitchClass
    is_minor = key.mode == 'minor'
    
    print(f"Generating chord {chord_type} in key {key} (minor={is_minor})")
    
    if is_minor:
        # Minor key chord patterns (i, ii°, III, iv, V, VI, VII)
        chord_patterns = {
            0: [0, 3, 7],     # i (minor tonic)
            1: [2, 5, 8],     # ii° (diminished)
            2: [3, 7, 10],    # III (major)
            3: [5, 8, 0],     # iv (minor)
            4: [7, 11, 2],    # V (major - dominant)
            5: [8, 0, 3],     # VI (major)
            6: [10, 2, 5],    # VII (major)
        }
    else:
        # Major key chord patterns (I, ii, iii, IV, V, vi, vii°)
        chord_patterns = {
            0: [0, 4, 7],     # I
            1: [2, 5, 9],     # ii
            2: [4, 7, 11],    # iii
            3: [5, 9, 0],     # IV
            4: [7, 11, 2],    # V
            5: [9, 0, 4],     # vi
            6: [11, 2, 5],    # vii°
        }
    
    pattern = chord_patterns.get(chord_type, [0, 3, 7] if is_minor else [0, 4, 7])
    
    # Convert to actual MIDI notes
    midi_notes = []
    for scale_degree in pattern:
        actual_pitch_class = (tonic + scale_degree) % 12
        midi_note = actual_pitch_class + 48  # Bass register
        
        # Avoid note collisions
        if len(midi_notes) > 0 and midi_note <= midi_notes[-1]:
            midi_note += 12
        
        midi_notes.append(midi_note)
    
    return midi_notes

def piano_roll_to_midi_chords(input_path, chord_probs, key, output_path):
    """Generate harmony with chords only on changes, rests otherwise"""
    
    # Load original MIDI to preserve exact melody
    original_mid = mido.MidiFile(input_path)
    
    # Create new MIDI file
    new_mid = mido.MidiFile(ticks_per_beat=original_mid.ticks_per_beat)
    
    # Copy original melody track(s)
    for i, track in enumerate(original_mid.tracks):
        new_track = mido.MidiTrack()
        for msg in track:
            new_track.append(msg.copy())
        new_mid.tracks.append(new_track)
        print(f"Copied original track {i}")
    
    print(f"Generating harmony for {len(chord_probs)} beats (beat-level changes)")
    
    # Get chord sequence
    predicted_chords = [np.argmax(chord_probs[i]) for i in range(len(chord_probs))]
    print(f"First 16 predicted chords: {predicted_chords[:16]}")
    
    # Add harmony track
    harmony_track = mido.MidiTrack()
    harmony_track.append(mido.Message('program_change', channel=1, program=0, time=0))
    
    ticks_per_beat = new_mid.ticks_per_beat
    
    for beat, chord_idx in enumerate(predicted_chords):
        # Check if this is a new chord (first beat or chord changed)
        is_new_chord = (beat == 0) or (chord_idx != predicted_chords[beat-1])
        
        if is_new_chord:
            print(f"Beat {beat+1}: Chord {chord_idx}")
            
            # Get chord notes
            chord_notes = chord_to_midi_notes(chord_idx, key)
            print(f"  Changed to chord {chord_idx}: {chord_notes}")
            
            # Play chord (all notes at once)
            for i, note in enumerate(chord_notes):
                harmony_track.append(mido.Message(
                    'note_on', channel=1, note=note, velocity=64, 
                    time=0  # All notes start together
                ))
            
            # Stop chord after one beat
            for i, note in enumerate(chord_notes):
                harmony_track.append(mido.Message(
                    'note_off', channel=1, note=note, velocity=0, 
                    time=ticks_per_beat if i == 0 else 0  # Stop after one beat
                ))
        else:
            print(f"Beat {beat+1}: Chord {chord_idx}")
            # Rest - just advance time without playing anything
            harmony_track.append(mido.Message(
                'control_change', channel=1, control=7, value=0, 
                time=ticks_per_beat  # Silent message to advance time
            ))
    
    # Add harmony track to MIDI
    new_mid.tracks.append(harmony_track)
    
    # Save the new MIDI file
    new_mid.save(output_path)
    print(f"Saved harmonized MIDI with {len(new_mid.tracks)} tracks")