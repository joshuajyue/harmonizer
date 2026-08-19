"""Hear it. `python -m ml.reharm.demo --tune twinkle --seeds 3 --midi out.mid`

Metrics are how this package is steered, but the goal is musical interest, and
nobody can hear a table. This prints the chord symbols of the skeleton and of
each reharmonization side by side with the substitution that produced them, and
writes a standard MIDI file so the result can actually be played.

The MIDI writer is thirty lines of `struct` rather than a dependency. The ml
package's requirements are shared with another workstream and a demo script is
not a good reason to add to them.
"""

from __future__ import annotations

import argparse
import struct
from collections.abc import Sequence
from pathlib import Path

from contracts.schema import Melody, Voice

from .engine import JAZZ_REHARM, JAZZ_REHARM_RULES
from .melodies import TRADITIONAL, jazz_tunes
from .metrics import score
from .model import ChordNGram
from .search import (
    HybridScorer,
    ReharmConfig,
    RuleScorer,
    build_lattice,
    realize,
    sample,
    viterbi,
)
from .skeleton import skeleton_from_rules

# ---------------------------------------------------------------------------
# A minimal standard MIDI file writer
# ---------------------------------------------------------------------------

TICKS_PER_BEAT = 480


def _variable_length(value: int) -> bytes:
    buffer = bytearray([value & 0x7F])
    value >>= 7
    while value:
        buffer.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(buffer))


def _track(events: Sequence[tuple[int, bytes]]) -> bytes:
    payload = bytearray()
    previous = 0
    for tick, message in sorted(events, key=lambda item: item[0]):
        payload += _variable_length(max(0, tick - previous)) + message
        previous = tick
    payload += _variable_length(0) + b"\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(payload)) + bytes(payload)


def write_midi(voices: Sequence[Voice], path: Path, *, tempo: float = 120.0) -> Path:
    """One track per voice, plus a tempo track. Format 1."""
    microseconds = int(60_000_000 / max(1.0, tempo))
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(voices) + 1, TICKS_PER_BEAT)
    tempo_track = _track([(0, b"\xff\x51\x03" + struct.pack(">I", microseconds)[1:])])

    tracks = [tempo_track]
    for index, voice in enumerate(voices):
        channel = index if index < 9 else index + 1  # skip the drum channel
        events: list[tuple[int, bytes]] = []
        for note in voice.notes:
            start = int(round(note.start * TICKS_PER_BEAT))
            stop = start + max(1, int(round(note.duration * TICKS_PER_BEAT)))
            pitch = max(0, min(127, int(note.pitch)))
            events.append((start, bytes([0x90 | channel, pitch, note.velocity])))
            events.append((stop, bytes([0x80 | channel, pitch, 0])))
        tracks.append(_track(events))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"".join(tracks))
    return path


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def describe(spans) -> str:
    parts = []
    for span in spans:
        text = span.chord.symbol()
        if span.chord.substitution_kind and span.chord.substitution_kind != "extension":
            text += f" <{span.chord.substitution_kind} for {span.chord.substitution_of}>"
        parts.append(text)
    return " | ".join(parts)


def run(melody: Melody, name: str, *, seeds: int, temperature: float, adventure: float, midi: Path | None) -> None:
    model = ChordNGram.load()
    if model is None:
        raise SystemExit("no model: run `python -m ml.reharm.model` first")

    config = ReharmConfig(adventure=adventure, temperature=temperature)
    skeleton = skeleton_from_rules(melody)
    base = skeleton.progression()
    lattice = build_lattice(skeleton, config)

    print("=" * 78)
    print(f"{name}   key {skeleton.key.tonic} {skeleton.mode}   {len(skeleton.units)} harmonic units")
    print("=" * 78)
    print("\nSKELETON (the rules engine, unreharmonized)")
    print("  " + " | ".join(span.chord.symbol() for span in base.spans))

    rule_result = realize(lattice, viterbi(lattice, RuleScorer(lattice, config)), skeleton)
    print("\nRULES (hand-written vocabulary, Viterbi argmax)")
    print("  " + describe(rule_result.spans))
    _show_score(skeleton, base, rule_result)

    scorer = HybridScorer(lattice, model, config)
    for index in range(seeds):
        result = realize(
            lattice,
            sample(lattice, scorer, temperature=temperature, top_p=config.top_p, seed=index + 1),
            skeleton,
        )
        print(f"\nSAMPLED seed={index + 1} (learned model, T={temperature})")
        print("  " + describe(result.spans))
        _show_score(skeleton, base, result)

    if midi is not None:
        for engine, suffix, temp, seed in (
            (JAZZ_REHARM_RULES, "rules", 0.0, None),
            (JAZZ_REHARM, "sampled", temperature, 1),
        ):
            harmonized = engine.harmonize(melody, voice_count=5, temperature=temp, seed=seed)
            path = midi.with_name(f"{midi.stem}_{suffix}{midi.suffix or '.mid'}")
            write_midi(harmonized.voices, path, tempo=melody.tempo)
            print(f"\nwritten: {path}")


def _show_score(skeleton, base, result) -> None:
    values = score(skeleton.melody, base, result.progression()).as_dict()
    print(
        f"    melody conflicts {values['hard_conflict_rate']:.3f}   "
        f"roots changed {values['root_change_rate']:.3f}   "
        f"chromatic {values['chromatic_tone_rate']:.3f}   "
        f"resolution {values['dominant_resolution_rate']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show and hear a reharmonization.")
    parser.add_argument("--tune", default="twinkle", help=f"one of {sorted(TRADITIONAL)} or jazz:<n>")
    parser.add_argument("--seeds", type=int, default=3)
    # One source of truth for the defaults: the engine's own config, so the demo
    # cannot drift into demonstrating a setting nobody ships.
    parser.add_argument("--temperature", type=float, default=ReharmConfig().temperature)
    parser.add_argument("--adventure", type=float, default=ReharmConfig().adventure)
    parser.add_argument("--midi", type=str, default=None, help="write MIDI files to this path")
    args = parser.parse_args()

    if args.tune.startswith("jazz:"):
        index = int(args.tune.split(":", 1)[1])
        tunes = jazz_tunes(limit=index + 1)
        if index >= len(tunes):
            raise SystemExit(f"only {len(tunes)} jazz tunes available")
        melody, name = tunes[index].melody, tunes[index].name
    else:
        if args.tune not in TRADITIONAL:
            raise SystemExit(f"unknown tune {args.tune!r}; try {sorted(TRADITIONAL)}")
        melody, name = TRADITIONAL[args.tune], args.tune

    run(
        melody,
        name,
        seeds=args.seeds,
        temperature=args.temperature,
        adventure=args.adventure,
        midi=Path(args.midi) if args.midi else None,
    )


if __name__ == "__main__":
    main()
