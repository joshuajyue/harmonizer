"""Substitution generation: does each device do what it is named after?

Every generator here encodes a piece of theory that has a right answer, so each
one is tested against that answer rather than against its own output. The
property tests at the end are the important ones for the product: whatever the
generators produce, it must never include a chord that cannot support the
melody sounding over it, because filtering that afterwards is precisely the
failure mode this package exists to avoid.
"""

import pytest

from ml.reharm.chords import SUBSTITUTION_KINDS, JazzChord
from ml.reharm.melodies import TRADITIONAL
from ml.reharm.metrics import note_weight
from ml.reharm.skeleton import Unit, skeleton_from_rules
from ml.reharm.substitutions import (
    MAX_HARD_CONFLICT,
    Context,
    backdoor_candidates,
    generate,
    identity_candidates,
    melody_penalty,
    modal_interchange_candidates,
    passing_dim_candidates,
    related_ii_candidates,
    secondary_dominant_candidates,
    tritone_candidates,
    upgrade_qualities,
)

C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B = range(12)


def unit(chord: JazzChord, *, duration: float = 4.0, melody=(), roman: str = "V") -> Unit:
    return Unit(start=0.0, duration=duration, base=chord, base_roman=roman, melody=list(melody), metric_level=3)


def context(following: JazzChord | None = None, *, tonic: int = C, mode: str = "major", **kwargs) -> Context:
    return Context(tonic=tonic, mode=mode, previous=None, following=following, **kwargs)


def roots(candidates) -> set[tuple[int, str]]:
    return {(chord.root, chord.quality) for candidate in candidates for chord in candidate.chords}


# ---------------------------------------------------------------------------
# Tritone substitution
# ---------------------------------------------------------------------------


def test_tritone_substitutes_v_with_bii7_in_c():
    candidates = tritone_candidates(
        unit(JazzChord(root=G, quality="dom7")),
        context(JazzChord(root=C, quality="maj7")),
    )
    assert (Db, "dom7") in roots(candidates)
    assert (D, "dom7") not in roots(candidates)


def test_tritone_is_not_offered_for_a_non_dominant():
    candidates = tritone_candidates(
        unit(JazzChord(root=D, quality="min7")),
        context(JazzChord(root=G, quality="dom7")),
    )
    assert candidates == []


def test_tritone_candidates_carry_their_provenance():
    candidates = tritone_candidates(
        unit(JazzChord(root=G, quality="dom7"), roman="V"),
        context(JazzChord(root=C, quality="maj7")),
    )
    assert candidates
    for candidate in candidates:
        assert candidate.kind == "tritone"
        substituted = [chord for chord in candidate.chords if chord.substitution_kind == "tritone"]
        assert substituted and all(chord.substitution_of == "V" for chord in substituted)
        # The substitute is the chord that was substituted IN, never the tune's
        # own chord sitting in front of it.
        assert all(chord.quality == "dom7" for chord in substituted)


def test_an_approach_split_only_tags_the_approach_chord():
    """"Dm7 | Db7" is a tritone sub of the Db7; the Dm7 is not a tritone of anything."""
    candidates = secondary_dominant_candidates(
        unit(JazzChord(root=C, quality="maj7"), duration=4.0, roman="I"),
        context(JazzChord(root=C, quality="maj7")),
    )
    for candidate in candidates:
        if len(candidate.chords) < 2:
            continue
        head = candidate.chords[0]
        if head.root == C:
            assert head.substitution_kind is None


# ---------------------------------------------------------------------------
# Secondary dominants and the two-bar ii-V
# ---------------------------------------------------------------------------


def test_secondary_dominant_targets_the_next_chord():
    candidates = secondary_dominant_candidates(
        unit(JazzChord(root=F, quality="maj7")),
        context(JazzChord(root=D, quality="min7")),
    )
    assert (A, "dom7") in roots(candidates)  # V7/ii


def test_related_ii_prepares_the_dominant_two_units_ahead():
    """This unit becomes the ii of a V that the NEXT unit can become."""
    candidates = related_ii_candidates(
        unit(JazzChord(root=C, quality="maj7")),
        context(JazzChord(root=F, quality="maj7"), following2=JazzChord(root=C, quality="maj7")),
    )
    assert (D, "min7") in roots(candidates)


def test_related_ii_of_a_minor_target_is_half_diminished():
    candidates = related_ii_candidates(
        unit(JazzChord(root=C, quality="min7")),
        context(
            JazzChord(root=F, quality="min7"),
            tonic=A,
            mode="minor",
            following2=JazzChord(root=A, quality="min7"),
        ),
    )
    assert (B, "halfdim7") in roots(candidates)


# ---------------------------------------------------------------------------
# Backdoor, mixture, passing diminished
# ---------------------------------------------------------------------------


def test_backdoor_into_the_tonic_is_bvii7():
    candidates = backdoor_candidates(
        unit(JazzChord(root=F, quality="maj7")),
        context(JazzChord(root=C, quality="maj7")),
    )
    assert (Bb, "dom7") in roots(candidates)


def test_backdoor_is_only_offered_into_a_tonic():
    candidates = backdoor_candidates(
        unit(JazzChord(root=F, quality="maj7")),
        context(JazzChord(root=D, quality="min7")),
    )
    assert candidates == []


def test_modal_interchange_borrows_the_minor_subdominant():
    candidates = modal_interchange_candidates(
        unit(JazzChord(root=F, quality="maj")),
        context(JazzChord(root=C, quality="maj7")),
    )
    assert (F, "min7") in roots(candidates) or (F, "min6") in roots(candidates)


def test_passing_diminished_leads_by_semitone_into_the_next_chord():
    candidates = passing_dim_candidates(
        unit(JazzChord(root=C, quality="maj7"), duration=4.0),
        context(JazzChord(root=D, quality="min7")),
    )
    assert candidates
    for candidate in candidates:
        approach = candidate.chords[-1]
        assert approach.quality == "dim7"
        assert (D - approach.root) % 12 == 1


def test_short_units_are_never_split():
    """One chord per beat is agitation, not reharmonization."""
    assert passing_dim_candidates(
        unit(JazzChord(root=C, quality="maj7"), duration=2.0),
        context(JazzChord(root=D, quality="min7")),
    ) == []


# ---------------------------------------------------------------------------
# Dialect: turning triads into jazz chords
# ---------------------------------------------------------------------------


def test_dominant_degree_upgrades_to_a_seventh_not_a_major_seventh():
    """Gmaj7 in C is a chord in G. The V triad takes the diatonic flat seventh."""
    options = upgrade_qualities(JazzChord(root=G, quality="maj"), context())
    assert options[0] == "dom7"
    assert "maj7" not in options


def test_tonic_triad_upgrades_to_a_major_seventh():
    options = upgrade_qualities(JazzChord(root=C, quality="maj"), context())
    assert "maj7" in options and "dom7" not in options


def test_identity_candidate_always_exists():
    candidates = identity_candidates(unit(JazzChord(root=C, quality="maj")), context())
    assert candidates


# ---------------------------------------------------------------------------
# Hard constraints, as properties
# ---------------------------------------------------------------------------


def test_melody_penalty_is_a_share_of_weight():
    weighted = [(F, note_weight(0.0, 2.0))]
    hard, soft = melody_penalty(JazzChord(root=C, quality="maj7"), weighted)
    assert hard == pytest.approx(1.0)
    assert soft == pytest.approx(0.0)


@pytest.mark.parametrize("name", ["twinkle", "greensleeves", "shenandoah", "blues_riff"])
def test_generated_candidates_respect_the_melody(name):
    """The lattice may not contain a chord the melody cannot live over.

    Enforced during generation rather than filtered afterwards, so a sampler
    downstream never has to reject-sample — this test is what makes that claim
    true rather than aspirational.
    """
    skeleton = skeleton_from_rules(TRADITIONAL[name])
    units = skeleton.units
    for index, item in enumerate(units):
        ctx = Context(
            tonic=skeleton.tonic,
            mode=skeleton.mode,
            previous=units[index - 1].base if index else None,
            following=units[index + 1].base if index + 1 < len(units) else None,
            following2=units[index + 2].base if index + 2 < len(units) else None,
            is_last=index == len(units) - 1,
        )
        for candidate in generate(item, ctx):
            if candidate.is_identity:
                continue  # the base chord is never refused, whatever it does
            assert candidate.melody_penalty <= MAX_HARD_CONFLICT + 0.35 + 1e-9


@pytest.mark.parametrize("name", ["twinkle", "greensleeves"])
def test_every_substitution_records_what_it_replaced(name):
    skeleton = skeleton_from_rules(TRADITIONAL[name])
    units = skeleton.units
    for index, item in enumerate(units):
        ctx = Context(
            tonic=skeleton.tonic,
            mode=skeleton.mode,
            previous=None,
            following=units[index + 1].base if index + 1 < len(units) else None,
            following2=units[index + 2].base if index + 2 < len(units) else None,
        )
        for candidate in generate(item, ctx):
            if candidate.kind is None:
                continue
            tagged = [chord for chord in candidate.chords if chord.substitution_kind]
            assert tagged, f"{candidate.label()} claims kind {candidate.kind} but tags nothing"
            assert any(chord.substitution_kind == candidate.kind for chord in tagged)
            for chord in tagged:
                assert chord.substitution_kind in SUBSTITUTION_KINDS
                assert chord.substitution_of == item.base_roman
