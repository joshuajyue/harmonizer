# backend/midi_utils.py
"""Renders a predicted chord sequence into a MIDI accompaniment track."""
import mido
import numpy as np

# Diatonic triads by scale degree (0-6), as pitch-class offsets from the tonic.
MAJOR_CHORD_TONES = {
    0: [0, 4, 7],     # I
    1: [2, 5, 9],     # ii
    2: [4, 7, 11],    # iii
    3: [5, 9, 0],     # IV
    4: [7, 11, 2],    # V
    5: [9, 0, 4],     # vi
    6: [11, 2, 5],    # vii°
}
MINOR_CHORD_TONES = {
    0: [0, 3, 7],     # i
    1: [2, 5, 8],     # ii°
    2: [3, 7, 10],    # III
    3: [5, 8, 0],     # iv
    4: [7, 11, 2],    # V
    5: [8, 0, 3],     # VI
    6: [10, 2, 5],    # VII
}

BASS_REGISTER = 48  # MIDI note for pitch class 0 (C3)


def chord_to_midi_notes(chord_degree, key):
    """Convert a chord degree (0-6) to a list of MIDI note numbers in `key`."""
    tonic = key.tonic.pitchClass
    is_minor = key.mode == 'minor'
    table = MINOR_CHORD_TONES if is_minor else MAJOR_CHORD_TONES
    pattern = table.get(chord_degree, [0, 3, 7] if is_minor else [0, 4, 7])

    midi_notes = []
    for scale_degree in pattern:
        note = (tonic + scale_degree) % 12 + BASS_REGISTER
        if midi_notes and note <= midi_notes[-1]:
            note += 12  # Avoid collisions/inversions collapsing onto the previous note
        midi_notes.append(note)
    return midi_notes


def piano_roll_to_midi_chords(input_path, chord_probs, key, output_path):
    """Write a new MIDI file: original melody track(s) plus a generated chord track.

    A new chord is only played when it changes from the previous beat; otherwise
    the harmony track stays silent for that beat.
    """
    original_mid = mido.MidiFile(input_path)
    new_mid = mido.MidiFile(ticks_per_beat=original_mid.ticks_per_beat)

    for track in original_mid.tracks:
        new_track = mido.MidiTrack()
        for msg in track:
            new_track.append(msg.copy())
        new_mid.tracks.append(new_track)

    predicted_chords = [int(np.argmax(chord_probs[i])) for i in range(len(chord_probs))]

    harmony_track = mido.MidiTrack()
    harmony_track.append(mido.Message('program_change', channel=1, program=0, time=0))
    ticks_per_beat = new_mid.ticks_per_beat

    for beat, chord_degree in enumerate(predicted_chords):
        is_new_chord = (beat == 0) or (chord_degree != predicted_chords[beat - 1])

        if is_new_chord:
            chord_notes = chord_to_midi_notes(chord_degree, key)
            for note in chord_notes:
                harmony_track.append(mido.Message('note_on', channel=1, note=note, velocity=64, time=0))
            for i, note in enumerate(chord_notes):
                harmony_track.append(mido.Message(
                    'note_off', channel=1, note=note, velocity=0,
                    time=ticks_per_beat if i == 0 else 0,
                ))
        else:
            # Silent message to advance the clock by one beat without re-triggering the chord.
            harmony_track.append(mido.Message('control_change', channel=1, control=7, value=0, time=ticks_per_beat))

    new_mid.tracks.append(harmony_track)
    new_mid.save(output_path)
