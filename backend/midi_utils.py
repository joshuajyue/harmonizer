# backend/midi_utils.py
import mido
import numpy as np

def midi_to_piano_roll(midi_file, sequence_length=32):
    """Convert MIDI to piano roll with 13 features (12 pitch classes + beat)"""
    try:
        mid = mido.MidiFile(midi_file)
        
        # Create feature matrix: 12 pitch classes + 1 beat feature
        features = np.zeros((sequence_length, 13))
        
        current_time = 0
        time_step = 0
        
        for track in mid.tracks:
            for msg in track:
                current_time += msg.time
                if time_step >= sequence_length:
                    break
                    
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Convert MIDI note to pitch class (0-11)
                    pitch_class = msg.note % 12
                    features[time_step, pitch_class] = 1
                    
                    # Simple beat detection (every 4th step is strong beat)
                    features[time_step, 12] = 1 if time_step % 4 == 0 else 0
                    
                if msg.time > 0:
                    time_step += 1
        
        return features
    except Exception as e:
        print(f"Error processing MIDI: {e}")
        return np.zeros((sequence_length, 13))

def piano_roll_to_midi(melody_roll, harmony_roll, output_file, tempo=120):
    """Convert melody and harmony piano rolls to MIDI with separate tracks"""
    mid = mido.MidiFile()
    
    # Track 1: Melody
    melody_track = mido.MidiTrack()
    mid.tracks.append(melody_track)
    melody_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
    melody_track.append(mido.Message('program_change', program=0))  # Piano
    
    # Track 2: Harmony
    harmony_track = mido.MidiTrack()
    mid.tracks.append(harmony_track)
    harmony_track.append(mido.Message('program_change', program=0))  # Piano
    
    ticks_per_beat = mid.ticks_per_beat
    time_per_step = ticks_per_beat // 2  # Eighth notes instead of 16th
    
    # Process melody track
    for step, notes in enumerate(melody_roll):
        active_pitch_classes = np.where(notes > 0.5)[0]
        
        for pitch_class in active_pitch_classes:
            midi_note = int(pitch_class) + 72  # Higher octave for melody (C5)
            
            # Note on
            melody_track.append(mido.Message('note_on', note=midi_note, 
                                           velocity=80, time=0))
            # Note off (longer duration)
            melody_track.append(mido.Message('note_off', note=midi_note, 
                                           velocity=80, time=time_per_step * 2))
    
    # Process harmony track
    for step, notes in enumerate(harmony_roll):
        active_pitch_classes = np.where(notes > 0.5)[0]
        
        for pitch_class in active_pitch_classes:
            midi_note = int(pitch_class) + 60  # Lower octave for harmony (C4)
            
            # Note on
            harmony_track.append(mido.Message('note_on', note=midi_note, 
                                            velocity=60, time=0))
            # Note off (longer duration)
            harmony_track.append(mido.Message('note_off', note=midi_note, 
                                            velocity=60, time=time_per_step * 2))
    
    mid.save(output_file)