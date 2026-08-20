"""Searches for an SATB voicing of the fixture containing exactly the intended defects.

Hand-voicing the fixture kept introducing accidental parallels: fixing one segment
created another two segments away. This does it properly — a DP over chord-tone
assignments per segment, scored by the real detector in ml.theory.voicing, with the
three intended defects pinned in place.

Run:  ./ml/.venv/bin/python contracts/examples/solve_voicing.py
It prints a PROGRESSION table to paste into build_fixtures.py.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.theory.voicing import find_parallels, find_voice_crossings  # noqa: E402

RANGES = {"S": (60, 79), "A": (55, 74), "T": (48, 67), "B": (40, 60)}

CHORDS = {
    "I": (0, "maj", [0, 4, 7]),
    "IV": (5, "maj", [5, 9, 0]),
    "V": (7, "maj", [7, 11, 2]),
}

# (start, duration, chord, soprano) — soprano is the melody and is immovable.
SEGMENTS = [
    (0, 2, "I", 60), (2, 2, "I", 64), (4, 2, "V", 67), (6, 2, "I", 64),
    (8, 1, "IV", 65), (9, 1, "I", 64), (10, 2, "V", 62), (12, 4, "I", 60),
    (16, 2, "I", 64), (18, 2, "V", 67), (20, 2, "IV", 69), (22, 2, "V", 67),
    (24, 1, "IV", 65), (25, 1, "I", 64), (26, 2, "V", 62), (28, 4, "I", 60),
]

# Pinned voicings that create the three intended defects. Index into SEGMENTS.
PINNED = {
    4: (65, 57, 48, 41),   # beat 8  IV  -> sets up parallel fifths
    5: (64, 60, 55, 48),   # beat 9  I   -> tenor/bass both +7, fifth to fifth
    10: (69, 60, 53, 45),  # beat 20 IV6 -> sets up parallel octaves
    11: (67, 59, 50, 43),  # beat 22 V   -> soprano/bass both -2, octave to octave
    13: (64, 55, 60, 48),  # beat 25 I   -> tenor (60) crosses above alto (55)
}

INTENDED = {
    (9.0, "parallel_fifths"),
    (22.0, "parallel_octaves"),
    (25.0, "voice_crossing"),
}


def candidates(chord: str, soprano: int) -> list[tuple[int, int, int, int]]:
    """All (S, A, T, B) assignments using only chord tones, in range, complete.

    Style constraints, without which the search happily returns technically clean but
    musically absurd voicings (it opened the piece on a second-inversion I64):
      - bass takes the root or the third only; no second inversions
      - strict ordering between inner voices, so no alto/tenor unisons
      - standard upper-voice spacing limit of an octave
    """
    _, _, tones = CHORDS[chord]
    root_pc, third_pc = tones[0], tones[1]
    out = []
    pitches = {
        voice: [p for p in range(RANGES[voice][0], RANGES[voice][1] + 1) if p % 12 in tones]
        for voice in ("A", "T", "B")
    }
    for alto, tenor, bass in product(pitches["A"], pitches["T"], pitches["B"]):
        if bass % 12 not in (root_pc, third_pc):
            continue
        if not (soprano >= alto > tenor > bass):
            continue
        if len({p % 12 for p in (soprano, alto, tenor, bass)}) != 3:
            continue  # require a complete triad
        if soprano - alto > 12 or alto - tenor > 12:
            continue
        out.append((soprano, alto, tenor, bass))
    return out


def inversion_penalty(chord: str, bass: int) -> int:
    """Prefer root position; first inversion is fine but should not be the default."""
    root_pc = CHORDS[chord][2][0]
    return 0 if bass % 12 == root_pc else 6


def defects_between(prev, curr, beat: float) -> set[tuple[float, str]]:
    return {(beat, kind) for _, _, kind in find_parallels(prev, curr)}


def solve() -> list[tuple]:
    options: list[list[tuple[int, int, int, int]]] = []
    for index, (_, _, chord, soprano) in enumerate(SEGMENTS):
        if index in PINNED:
            options.append([PINNED[index]])
        else:
            found = candidates(chord, soprano)
            if not found:
                raise SystemExit(f"segment {index}: no candidate voicings")
            options.append(found)

    # DP: state is the chosen voicing of the current segment.
    best: dict[tuple, tuple[int, list]] = {}
    for voicing in options[0]:
        crossing = {(float(SEGMENTS[0][0]), "voice_crossing")} if find_voice_crossings(voicing) else set()
        penalty = len(crossing - INTENDED)
        best[voicing] = (penalty, [voicing])

    for index in range(1, len(SEGMENTS)):
        beat = float(SEGMENTS[index][0])
        nxt: dict[tuple, tuple[int, list]] = {}
        for voicing in options[index]:
            crossing = {(beat, "voice_crossing")} if find_voice_crossings(voicing) else set()
            best_cost, best_path = None, None
            for prev_voicing, (cost, path) in best.items():
                found = defects_between(prev_voicing, voicing, beat) | crossing
                added = len(found - INTENDED) + len(INTENDED & found) * 0  # unintended only
                # Penalise motion that is musically poor: large leaps in inner voices,
                # and inversions where root position would serve.
                leap = sum(abs(voicing[v] - prev_voicing[v]) for v in (1, 2, 3))
                style = inversion_penalty(SEGMENTS[index][2], voicing[3])
                total = cost + added * 1000 + leap + style
                if best_cost is None or total < best_cost:
                    best_cost, best_path = total, path + [voicing]
            nxt[voicing] = (best_cost, best_path)
        best = nxt

    cost, path = min(best.values(), key=lambda item: item[0])
    print(f"search complete, cost={cost} (>=1000 means an unintended defect remains)")
    return path


def main() -> None:
    path = solve()
    print()
    for (start, duration, chord, _), voicing in zip(SEGMENTS, path):
        root, quality, tones = CHORDS[chord]
        # `tones` holds absolute pitch classes, so index the bass pitch class directly.
        bass_pc = voicing[3] % 12
        inversion = tones.index(bass_pc) if bass_pc in tones else 0
        roman = chord + ("6" if inversion == 1 else "64" if inversion == 2 else "")
        print(
            f'    ({start:<2}, {duration}, "{roman}", {root}, "{quality}", {inversion}, '
            f"{voicing}),"
        )


if __name__ == "__main__":
    main()
