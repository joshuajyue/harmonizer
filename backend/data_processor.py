# backend/data_processor.py
"""Feature extraction and rule-based chord selection for the creative engine.

Melody is sampled one feature vector per quarter-note beat. Each beat vector is:
  - 12 dims: active pitch classes (one-hot, multiple can be set for double-stops)
  - 1 dim:   strong-beat indicator (downbeat of the measure)
  - 1 dim:   key mode indicator (1 = minor, 0 = major)
Chords are chosen per measure (one chord type 0-6, diatonic scale degree) and
expanded back out to beat-level labels.
"""
import music21
import numpy as np

FEATURE_DIM = 14
NUM_CHORD_TYPES = 7
SEQUENCE_LENGTH = 32

# Diatonic triads by scale degree (0-6), as pitch-class offsets from the tonic.
MAJOR_CHORD_TONES = {
    0: [0, 4, 7],    # I
    1: [2, 5, 9],    # ii
    2: [4, 7, 11],   # iii
    3: [5, 9, 0],    # IV
    4: [7, 11, 2],   # V
    5: [9, 0, 4],    # vi
    6: [11, 2, 5],   # vii°
}
MINOR_CHORD_TONES = {
    0: [0, 3, 7],    # i
    1: [2, 5, 8],    # ii°
    2: [3, 7, 10],   # III
    3: [5, 8, 0],    # iv
    4: [7, 11, 2],   # V
    5: [8, 0, 3],    # VI
    6: [10, 2, 5],   # VII
}

# Common functional progressions (from_degree, to_degree) -> bonus applies.
MAJOR_FUNCTIONAL_PROGRESSIONS = {
    (0, 4), (0, 3), (0, 5), (0, 1),
    (4, 0), (3, 0), (3, 4),
    (5, 3), (5, 4),
    (1, 4), (1, 0),
    (2, 5), (2, 1),
    (6, 0), (6, 4),
}
MINOR_FUNCTIONAL_PROGRESSIONS = {
    (0, 3), (0, 4), (0, 5),
    (3, 0), (3, 4),
    (4, 0), (4, 5),
    (5, 3), (5, 4),
    (1, 4), (2, 5),
    (6, 0), (6, 4),
}


class MeasureBasedChordProcessor:
    """Extracts melody features and generates rule-based chord progressions."""

    def get_chord_tones(self, chord_degree, is_minor=False):
        """Pitch-class offsets (from tonic) for a diatonic triad."""
        table = MINOR_CHORD_TONES if is_minor else MAJOR_CHORD_TONES
        return table.get(chord_degree, [0, 3, 7] if is_minor else [0, 4, 7])

    def is_neighbor_tone(self, pitch, chord_tones):
        """Whether `pitch` sits a half-step from any chord tone."""
        return any(abs(pitch - tone) in (1, 11) for tone in chord_tones)

    def is_functional_progression(self, chord1, chord2, is_minor):
        progressions = MINOR_FUNCTIONAL_PROGRESSIONS if is_minor else MAJOR_FUNCTIONAL_PROGRESSIONS
        return (chord1, chord2) in progressions

    def detect_key(self, score):
        """Detect the key of a music21 score, defaulting to C major on failure."""
        try:
            return score.analyze('key')
        except Exception:
            return music21.key.Key('C', 'major')

    def find_chord_for_melody_context_pitches(self, melody_pitch_classes, key, prev_chord):
        """Pick the best-fitting diatonic chord degree for a set of melody pitch classes."""
        if not melody_pitch_classes:
            return 0  # Default to tonic when no notes sound in the measure

        tonic = key.tonic.pitchClass
        is_minor = key.mode == 'minor'
        scale_degrees = [(pc - tonic) % 12 for pc in melody_pitch_classes]

        chord_scores = {}
        for chord_degree in range(NUM_CHORD_TYPES):
            chord_tones = self.get_chord_tones(chord_degree, is_minor)
            score = 0
            for degree in scale_degrees:
                if degree in chord_tones:
                    score += 3  # Chord tone: strong fit
                elif self.is_neighbor_tone(degree, chord_tones):
                    score += 1  # Passing/neighbor tone
                else:
                    score -= 1  # Dissonant against this chord

            if prev_chord is not None and self.is_functional_progression(prev_chord, chord_degree, is_minor):
                score += 2
            elif prev_chord is None and chord_degree in (0, 3, 4):  # I/IV/V is a safe opening chord
                score += 1

            chord_scores[chord_degree] = score

        return max(chord_scores, key=chord_scores.get)

    def generate_creative_chord_progression(self, melody_measures, key, num_measures):
        """Generate one chord degree per measure via greedy best-fit selection."""
        measure_chords = []
        for measure_idx in range(num_measures):
            prev_chord = measure_chords[-1] if measure_chords else None
            best_chord = self.find_chord_for_melody_context_pitches(
                melody_measures[measure_idx], key, prev_chord
            )
            measure_chords.append(best_chord)
        return measure_chords

    def extract_measure_based_features(self, melody_part, key):
        """Extract one feature vector per quarter-note beat, plus per-beat chord labels.

        Returns (features, chord_labels, actual_length) where features/chord_labels
        are padded or truncated to SEQUENCE_LENGTH, and actual_length is the true
        (unpadded) number of beats in the piece.
        """
        time_sig = melody_part.getTimeSignatures()[0] if melody_part.getTimeSignatures() else music21.meter.TimeSignature('4/4')
        beats_per_measure = time_sig.numerator
        total_quarters = int(melody_part.duration.quarterLength)

        is_minor = 1 if (key and key.mode == 'minor') else 0
        # .offset on a recursed element is relative to its immediate parent (e.g. resets every
        # measure); getOffsetInHierarchy gives the absolute position needed for beat windowing.
        notes = [
            (note, note.getOffsetInHierarchy(melody_part), note.duration.quarterLength)
            for note in melody_part.recurse().notes
        ]

        features = []
        for quarter_beat in range(total_quarters):
            beat_start, beat_end = quarter_beat, quarter_beat + 1

            pitch_class_vector = np.zeros(12)
            for note, note_start, note_duration in notes:
                note_end = note_start + note_duration
                if note_start < beat_end and note_end > beat_start:
                    if hasattr(note, 'pitch'):
                        pitch_class_vector[note.pitch.pitchClass] = 1
                    elif hasattr(note, 'pitches'):
                        for p in note.pitches:
                            pitch_class_vector[p.pitchClass] = 1

            strong_beat = 1 if (quarter_beat % beats_per_measure == 0) else 0
            features.append(np.concatenate([pitch_class_vector, [strong_beat], [is_minor]]))

        features = np.array(features) if features else np.zeros((0, FEATURE_DIM))
        actual_length = len(features)

        chord_progression = [0] * actual_length
        if actual_length > 0:
            # Group beats into measures, collect their pitch classes, and pick one chord per measure.
            measures = []
            for measure_start in range(0, actual_length, beats_per_measure):
                measure_end = min(measure_start + beats_per_measure, actual_length)
                measure_pitch_classes = set()
                for beat in range(measure_start, measure_end):
                    measure_pitch_classes.update(i for i, val in enumerate(features[beat][:12]) if val > 0)
                measures.append(list(measure_pitch_classes))

            measure_chords = self.generate_creative_chord_progression(measures, key, len(measures))

            chord_progression = []
            for measure_idx, measure_chord in enumerate(measure_chords):
                for beat_in_measure in range(beats_per_measure):
                    if len(chord_progression) >= actual_length:
                        break
                    # Alternate measures get a beat-3 move to the dominant, for gentle harmonic motion.
                    if beat_in_measure == 2 and measure_idx % 2 == 1 and measure_chord == 0:
                        chord_progression.append(4)
                    else:
                        chord_progression.append(measure_chord)

        chord_labels = np.zeros((actual_length, NUM_CHORD_TYPES))
        for i, chord in enumerate(chord_progression):
            chord_labels[i, chord] = 1

        if actual_length < SEQUENCE_LENGTH:
            feature_padding = np.zeros((SEQUENCE_LENGTH - actual_length, FEATURE_DIM))
            chord_padding = np.zeros((SEQUENCE_LENGTH - actual_length, NUM_CHORD_TYPES))
            chord_padding[:, 0] = 1  # Default padding to tonic
            features = np.vstack([features, feature_padding])
            chord_labels = np.vstack([chord_labels, chord_padding])
        else:
            features = features[:SEQUENCE_LENGTH]
            chord_labels = chord_labels[:SEQUENCE_LENGTH]

        return features, chord_labels, actual_length

    def extract_real_chord_labels(self, score, key, actual_length):
        """Derive per-beat chord-degree labels from the actual SATB harmony (not melody alone).

        Merges every part with `chordify()` and, for each quarter-note beat, matches the
        pitch classes Bach actually sounded against the closest diatonic triad. This is what
        makes the neural model learn genuine harmonic style instead of imitating the
        melody-only rule engine used by `extract_measure_based_features`.
        """
        tonic = key.tonic.pitchClass
        is_minor = key.mode == 'minor'
        chordified = score.chordify()
        harmony_notes = [
            (element, element.getOffsetInHierarchy(chordified), element.duration.quarterLength)
            for element in chordified.recurse().notes
        ]

        chord_degrees = []
        for quarter_beat in range(actual_length):
            beat_start, beat_end = quarter_beat, quarter_beat + 1

            pitch_classes = set()
            for element, note_start, note_duration in harmony_notes:
                note_end = note_start + note_duration
                if note_start < beat_end and note_end > beat_start:
                    if hasattr(element, 'pitches'):
                        pitch_classes.update(p.pitchClass for p in element.pitches)
                    elif hasattr(element, 'pitch'):
                        pitch_classes.add(element.pitch.pitchClass)

            if not pitch_classes:
                chord_degrees.append(chord_degrees[-1] if chord_degrees else 0)
                continue

            scale_degrees = [(pc - tonic) % 12 for pc in pitch_classes]
            best_degree, best_score = 0, float('-inf')
            for degree in range(NUM_CHORD_TYPES):
                chord_tones = self.get_chord_tones(degree, is_minor)
                score_val = sum(1 if d in chord_tones else -1 for d in scale_degrees)
                if score_val > best_score:
                    best_score, best_degree = score_val, degree
            chord_degrees.append(best_degree)

        return chord_degrees

    def process_bach_chorales(self, max_pieces=100):
        """Build a training set of (melody features, real chord labels) pairs from the Bach corpus.

        Inputs are melody-only features (what the model sees at inference time). Targets are
        the real chord Bach used in the full SATB harmony at each beat, so the model learns
        actual harmonic style rather than reproducing the rule-based creative engine.
        """
        training_data = []

        try:
            chorale_paths = music21.corpus.getComposer('bach')[:max_pieces]
        except Exception as e:
            print(f"Could not load Bach corpus: {e}")
            return []

        for chorale_path in chorale_paths:
            try:
                score = music21.corpus.parse(chorale_path)
                parts = list(score.parts)
                if not parts:
                    continue
                melody_part = parts[0]  # Soprano line

                key = self.detect_key(score)
                # extract_measure_based_features already pads/truncates `features` to
                # SEQUENCE_LENGTH; `actual_length` is the true (pre-padding) beat count.
                features, _, actual_length = self.extract_measure_based_features(melody_part, key)
                if actual_length < 8:
                    continue  # Skip pieces too short to be useful training examples

                real_chords = self.extract_real_chord_labels(score, key, actual_length)
                chord_labels = np.zeros((actual_length, NUM_CHORD_TYPES))
                for i, chord in enumerate(real_chords):
                    chord_labels[i, chord] = 1

                if actual_length < SEQUENCE_LENGTH:
                    chord_padding = np.zeros((SEQUENCE_LENGTH - actual_length, NUM_CHORD_TYPES))
                    chord_padding[:, 0] = 1
                    chord_labels = np.vstack([chord_labels, chord_padding])
                else:
                    chord_labels = chord_labels[:SEQUENCE_LENGTH]

                training_data.append({'input': features, 'target': chord_labels})
            except Exception as e:
                print(f"Skipping {chorale_path}: {e}")
                continue

        print(f"Processed {len(training_data)} Bach chorales into training examples")
        return training_data
