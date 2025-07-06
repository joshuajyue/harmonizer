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

def piano_roll_to_midi(piano_roll, output_file, tempo=120):
    """Convert piano roll back to MIDI (expecting 12 pitch classes)"""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
    
    ticks_per_beat = mid.ticks_per_beat
    time_per_step = ticks_per_beat // 4
    
    for step, notes in enumerate(piano_roll):
        # Find active pitch classes (probability > 0.5)
        if len(notes) == 12:  # Harmony output
            active_pitch_classes = np.where(notes > 0.5)[0]
        else:  # Combined output (13 features)
            active_pitch_classes = np.where(notes[:12] > 0.5)[0]
        
        for pitch_class in active_pitch_classes:
            # Convert pitch class back to MIDI note (use octave 4: C4=60)
            midi_note = int(pitch_class) + 60
            
            # Note on
            track.append(mido.Message('note_on', note=midi_note, 
                                    velocity=64, time=0))
            # Note off
            track.append(mido.Message('note_off', note=midi_note, 
                                    velocity=64, time=time_per_step))
    
    mid.save(output_file)