"""Metrics and corpora.

The metrics are what every claim in REPORT.md rests on, so their arithmetic is
pinned to hand-computed cases. The corpus tests are skipped when the cache is
absent — they need a 4 MB treebank and a 42 MB database that are deliberately
not committed — but when it is present they check the two things a corpus
loader gets silently wrong: reading a chord symbol as the wrong chord, and
losing the harmonic rhythm.
"""

import pytest

from ml.reharm.chords import JazzChord
from ml.reharm.data import (
    TREEBANK_PATH,
    WJAZZD_PATH,
    ChordSpan,
    Progression,
    parse_weimar_chord,
    treebank_progressions,
)
from ml.reharm.metrics import (
    DISTANCE_BAND,
    collect_syntax,
    diatonic_pcs,
    distance,
    distance_reward,
    js_divergence,
    melody_fit,
    note_weight,
    score,
)

C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B = range(12)


def progression(*chords_and_durations, tonic: int = C, mode: str = "major") -> Progression:
    spans = []
    offset = 0.0
    for chord, duration in chords_and_durations:
        spans.append(ChordSpan(offset, duration, chord))
        offset += duration
    return Progression(spans=spans, tonic=tonic, mode=mode)


II_V_I = progression(
    (JazzChord(root=D, quality="min7"), 4.0),
    (JazzChord(root=G, quality="dom7"), 4.0),
    (JazzChord(root=C, quality="maj7"), 8.0),
)


# ---------------------------------------------------------------------------
# Harmonic syntax
# ---------------------------------------------------------------------------


def test_ii_v_i_is_recognised_as_such():
    counts = collect_syntax(II_V_I)
    assert counts.ii_v == 1
    assert counts.ii_v_i == 1
    assert counts.dominants == 1
    assert counts.dominants_resolved == 1
    assert counts.dominants_down_fifth == 1


def test_tritone_substitution_still_counts_as_a_resolution():
    substituted = progression(
        (JazzChord(root=D, quality="min7"), 4.0),
        (JazzChord(root=Db, quality="dom7"), 4.0),
        (JazzChord(root=C, quality="maj7"), 8.0),
    )
    counts = collect_syntax(substituted)
    assert counts.dominants_resolved == 1
    assert counts.dominants_down_semitone == 1


def test_a_dominant_that_goes_nowhere_is_not_resolved():
    stranded = progression(
        (JazzChord(root=G, quality="dom7"), 4.0),
        (JazzChord(root=E, quality="maj7"), 4.0),
    )
    assert collect_syntax(stranded).dominants_resolved == 0


def test_chromatic_tones_are_counted_against_the_key():
    assert Gb not in diatonic_pcs(C, "major")
    counts = collect_syntax(progression((JazzChord(root=Db, quality="dom7"), 4.0)))
    assert counts.chromatic_tones > 0
    assert counts.nondiatonic_roots == 1


def test_seventh_and_extension_rates():
    stats = collect_syntax(II_V_I).as_dict()
    assert stats["seventh_rate"] == pytest.approx(1.0)
    assert stats["extension_rate"] == pytest.approx(0.0)
    assert stats["mean_chord_beats"] == pytest.approx(16.0 / 3)


# ---------------------------------------------------------------------------
# Melody compatibility
# ---------------------------------------------------------------------------


def test_melody_note_is_judged_against_every_chord_it_spans():
    """A held note over a chord change must be judged against both chords."""
    held = [(0.0, 65, 8.0)]  # F4 across the whole thing
    fit = melody_fit(held, progression(
        (JazzChord(root=F, quality="maj7"), 4.0),
        (JazzChord(root=C, quality="maj7"), 4.0),
    ))
    assert fit.chord_tone_rate == pytest.approx(0.5, abs=0.01)
    assert fit.hard_conflict_rate == pytest.approx(0.5, abs=0.01)


def test_strong_beats_weigh_more_than_offbeats():
    assert note_weight(0.0, 1.0) > note_weight(0.5, 1.0)
    assert note_weight(1.0, 1.0) > note_weight(1.5, 1.0)


def test_a_clean_melody_over_its_own_changes_has_no_conflicts():
    melody = [(0.0, 62, 4.0), (4.0, 71, 4.0), (8.0, 72, 8.0)]
    fit = melody_fit(melody, II_V_I)
    assert fit.hard_conflict_rate == pytest.approx(0.0)
    assert fit.chord_tone_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def test_distance_from_itself_is_zero():
    metrics = distance(II_V_I, II_V_I)
    assert metrics.changed_rate == 0.0
    assert metrics.root_change_rate == 0.0
    assert metrics.pc_distance == 0.0


def test_distance_survives_an_inserted_chord():
    """Time alignment, not index alignment: an insertion is local, not total."""
    with_insert = progression(
        (JazzChord(root=D, quality="min7"), 4.0),
        (JazzChord(root=Db, quality="dom7"), 2.0),
        (JazzChord(root=G, quality="dom7"), 2.0),
        (JazzChord(root=C, quality="maj7"), 8.0),
    )
    metrics = distance(II_V_I, with_insert)
    assert 0.0 < metrics.root_change_rate < 0.35


def test_distance_reward_peaks_inside_the_calibrated_band():
    low, high = DISTANCE_BAND
    assert distance_reward((low + high) / 2) == 1.0
    assert distance_reward(0.0) == 0.0
    assert distance_reward(1.0) < 0.5


def test_score_combines_the_three_axes():
    result = score([(0.0, 62, 4.0), (4.0, 71, 4.0), (8.0, 72, 8.0)], II_V_I, II_V_I)
    assert 0.0 <= result.headline <= 1.0
    assert result.melody_penalty == pytest.approx(0.0)


def test_js_divergence_bounds():
    assert js_divergence({}, {}) == 0.0
    assert js_divergence({"a": 10}, {"a": 10}) == pytest.approx(0.0, abs=1e-9)
    assert js_divergence({"a": 100}, {"b": 100}) > 0.5


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

needs_treebank = pytest.mark.skipif(
    not TREEBANK_PATH.exists(), reason="treebank not cached (python -m ml.reharm.oracle)"
)
needs_wjazzd = pytest.mark.skipif(
    not WJAZZD_PATH.exists(), reason="wjazzd not cached (python -m ml.reharm.oracle)"
)


@pytest.mark.parametrize(
    "text,root,quality,extensions",
    [
        ("Bb7", Bb, "dom7", ()),
        ("C-7", C, "min7", ()),
        ("Ebj7", Eb, "maj7", ()),
        ("F79b", F, "dom7", ("b9",)),
        ("Gj7911#", G, "maj7", ("9", "#11")),
        ("Absus7", Ab, "sus4", ("7",)),
        ("Am7b5", A, "halfdim7", ()),
        ("D7alt", D, "dom7", ("b9", "#9", "b13")),
        ("Eo7", E, "dim7", ()),
        ("B+7", B, "dom7", ("#5",)),
    ],
)
def test_weimar_chord_spelling(text, root, quality, extensions):
    """Weimar puts the accidental AFTER the degree: "79b" is 7 with a flat nine."""
    chord = parse_weimar_chord(text)
    assert chord is not None
    assert (chord.root, chord.quality) == (root, quality)
    assert set(chord.extensions) == set(extensions)


def test_weimar_no_chord_is_not_a_chord():
    assert parse_weimar_chord("NC") is None
    assert parse_weimar_chord("") is None


@needs_treebank
def test_treebank_loads_and_parses_completely():
    tunes = treebank_progressions(download=False)
    assert len(tunes) > 1000
    assert all(tune.spans for tune in tunes)
    assert all(0 <= tune.tonic <= 11 for tune in tunes)


@needs_treebank
def test_treebank_chord_positions_are_monotonic():
    for tune in treebank_progressions(download=False)[:50]:
        for previous, current in zip(tune.spans, tune.spans[1:]):
            assert current.start >= previous.start
            assert previous.duration > 0


@needs_wjazzd
def test_weimar_solo_keeps_its_harmonic_rhythm():
    """Weimar writes a chord only where it changes; holding it is our job."""
    from ml.reharm.data import chorus_progression, load_solos

    solos = load_solos(download=False, limit=3)
    assert solos
    chorus = chorus_progression(solos[0], 1)
    assert chorus.spans
    for previous, current in zip(chorus.spans, chorus.spans[1:]):
        assert current.start == pytest.approx(previous.stop)
    assert max(span.duration for span in chorus.spans) > 1.0
