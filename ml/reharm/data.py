"""Corpora: the Jazz Harmony Treebank and the Weimar Jazz Database.

Both are downloaded to a gitignored cache on first use and never committed.

  * **Jazz Harmony Treebank** (Harasim, Finkensiep, Ericson, O'Donnell,
    Rohrmeier, ISMIR 2020) — CC BY 4.0, https://github.com/DCMLab/JazzHarmonyTreebank.
    1170 chord sequences derived from the open iRealPro collection, 150 of them
    with hierarchical harmonic analyses. This is the harmonic-syntax corpus.

  * **Weimar Jazz Database** v2.1 (The Jazzomat Research Project, 2012-2017) —
    ODbL 1.0 / DbCL 1.0, https://jazzomat.hfm-weimar.de. 456 transcribed solos
    with beat-aligned chord changes. This is the *melody* corpus, and it is the
    only source here that pairs a real jazz line with the chords a real rhythm
    section actually played under it — which is exactly what a melody-harmony
    compatibility metric has to be calibrated against.

Raw iRealPro data is not openly licensed and is deliberately not used; the
treebank's derived, CC BY 4.0 release is.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .chords import JazzChord, parse_key, parse_symbol

CACHE = Path(__file__).resolve().parent / "cache"

TREEBANK_URL = "https://raw.githubusercontent.com/DCMLab/JazzHarmonyTreebank/master/treebank.json"
WJAZZD_URL = "https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db"

TREEBANK_PATH = CACHE / "treebank.json"
WJAZZD_PATH = CACHE / "wjazzd.db"

LICENCES = {
    "treebank": "Jazz Harmony Treebank (DCMLab) — CC BY 4.0",
    "wjazzd": "Weimar Jazz Database v2.1 (Jazzomat Research Project) — ODbL 1.0 / DbCL 1.0",
}


def _download(url: str, path: Path) -> Path:
    # urllib and sqlite3 are imported inside the functions that use them, not at
    # module scope. The backend imports every module in this package at startup
    # to discover engines, and the engine path never touches a corpus — so 60 ms
    # of import cost for a downloader that will not run is 60 ms of slower boot
    # for every process that only wants to harmonize a melody.
    import urllib.request

    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as response, tmp.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Progressions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChordSpan:
    """One chord sounding over a stretch of time, in quarter-note beats."""

    start: float
    duration: float
    chord: JazzChord

    @property
    def stop(self) -> float:
        return self.start + self.duration


@dataclass
class Progression:
    """A chord sequence in a key — the unit everything in this package works on."""

    spans: list[ChordSpan]
    tonic: int
    mode: str
    meter: tuple[int, int] = (4, 4)
    title: str = ""
    source: str = ""

    @property
    def chords(self) -> list[JazzChord]:
        return [span.chord for span in self.spans]

    @property
    def duration(self) -> float:
        return self.spans[-1].stop if self.spans else 0.0

    def chord_at(self, offset: float) -> JazzChord | None:
        for span in self.spans:
            if span.start <= offset < span.stop:
                return span.chord
        return None

    def transposed(self, semitones: int) -> Progression:
        from dataclasses import replace as _replace

        spans = [
            ChordSpan(
                span.start,
                span.duration,
                _replace(
                    span.chord,
                    root=(span.chord.root + semitones) % 12,
                    bass=None if span.chord.bass is None else (span.chord.bass + semitones) % 12,
                ),
            )
            for span in self.spans
        ]
        return Progression(spans, (self.tonic + semitones) % 12, self.mode, self.meter, self.title, self.source)


# ---------------------------------------------------------------------------
# Jazz Harmony Treebank
# ---------------------------------------------------------------------------


def load_treebank(*, path: Path = TREEBANK_PATH, download: bool = True) -> list[dict]:
    if download:
        _download(TREEBANK_URL, path)
    return json.loads(path.read_text())


def treebank_progressions(*, path: Path = TREEBANK_PATH, download: bool = True) -> list[Progression]:
    """Every treebank tune as a `Progression`, chord positions in beats.

    Tunes whose key or any chord fails to parse are skipped rather than guessed
    at: a silently mis-read chord pollutes the corpus statistics that the whole
    package is calibrated against.
    """
    out: list[Progression] = []
    for entry in load_treebank(path=path, download=download):
        key = parse_key(entry.get("key", ""))
        if key is None:
            continue
        numerator = int(entry["meter"]["numerator"])
        denominator = int(entry["meter"]["denominator"])
        beat_scale = 4.0 / denominator

        positions: list[float] = []
        chords: list[JazzChord] = []
        broken = False
        for measure, beat, symbol in zip(entry["measures"], entry["beats"], entry["chords"]):
            chord = parse_symbol(symbol)
            if chord is None:
                broken = True
                break
            positions.append(((int(measure) - 1) * numerator + (float(beat) - 1)) * beat_scale)
            chords.append(chord)
        if broken or not chords:
            continue

        end = positions[-1] + numerator * beat_scale
        spans = [
            ChordSpan(start, max(beat_scale, (positions[i + 1] if i + 1 < len(positions) else end) - start), chord)
            for i, (start, chord) in enumerate(zip(positions, chords))
        ]
        out.append(Progression(
            spans=spans,
            tonic=key[0],
            mode=key[1],
            meter=(numerator, denominator),
            title=entry.get("title", ""),
            source="treebank",
        ))
    return out


def treebank_trees(*, path: Path = TREEBANK_PATH, download: bool = True) -> list[tuple[str, dict]]:
    """The 150 hierarchical analyses, as (title, open_constituent_tree).

    The trees encode prolongation and cadential structure rather than mere chord
    bigrams: a chord's parent is the chord it is subordinate to. That is what
    makes it possible to ask which chords are *structural* — and structural
    chords are precisely the ones a reharmonization must not casually destroy.
    """
    out: list[tuple[str, dict]] = []
    for entry in load_treebank(path=path, download=download):
        for tree in entry.get("trees", []):
            root = tree.get("open_constituent_tree")
            if root:
                out.append((entry.get("title", ""), root))
    return out


def tree_depths(node: dict, depth: int = 0, acc: dict[int, int] | None = None) -> dict[int, int]:
    """Map leaf index -> depth of the leaf in the constituent tree."""
    if acc is None:
        acc = {}
    children = node.get("children") or []
    if not children:
        acc[len(acc)] = depth
        return acc
    for child in children:
        tree_depths(child, depth + 1, acc)
    return acc


# ---------------------------------------------------------------------------
# Weimar Jazz Database
# ---------------------------------------------------------------------------

#: Quality tokens in the Weimar chord spelling, longest first so that "-j7"
#: never gets read as "-" followed by junk.
_WEIMAR_QUALITIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("m7b5", "halfdim7", ()),
    ("sus7", "sus4", ("7",)),
    ("+j7", "maj7", ("#5",)),
    ("-j7", "minmaj7", ()),
    ("sus", "sus4", ()),
    ("j7", "maj7", ()),
    ("o7", "dim7", ()),
    ("+7", "dom7", ("#5",)),
    ("-7", "min7", ()),
    ("-6", "min6", ()),
    ("7", "dom7", ()),
    ("6", "maj6", ()),
    ("o", "dim", ()),
    ("+", "aug", ()),
    ("-", "min", ()),
    ("", "maj", ()),
)

_WEIMAR_ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_weimar_chord(text: str) -> JazzChord | None:
    """Parse the Weimar chord spelling, where tensions trail their accidental.

    "Bb79b13b" is Bb7(b9,b13) and "Ej7911#" is Emaj7(9,#11). The accidental
    comes AFTER the degree, which is the opposite of every other convention in
    this package, so it gets its own parser rather than a shared table.
    """
    if not text or text in ("NC", "N.C."):
        return None
    body = text.strip()

    bass: int | None = None
    if "/" in body:
        head, _, tail = body.rpartition("/")
        parsed_bass = _weimar_root(tail)
        if parsed_bass is not None and parsed_bass[1] == "" and head:
            bass, body = parsed_bass[0], head

    parsed = _weimar_root(body)
    if parsed is None:
        return None
    root, rest = parsed

    for token, quality, base_extensions in _WEIMAR_QUALITIES:
        if not rest.startswith(token):
            continue
        tail = rest[len(token):]
        extensions = list(base_extensions)
        if tail.startswith("alt"):
            return JazzChord(root, quality, tuple(extensions) + ("b9", "#9", "b13"), bass=bass)
        index = 0
        while index < len(tail):
            for degree in ("11", "13", "9"):
                if tail.startswith(degree, index):
                    index += len(degree)
                    accidental = ""
                    if index < len(tail) and tail[index] in "b#":
                        accidental = tail[index]
                        index += 1
                    extensions.append(accidental + degree)
                    break
            else:
                return None  # unrecognised trailing text: refuse rather than guess
        return JazzChord(root, quality, tuple(dict.fromkeys(extensions)), bass=bass)
    return None


def _weimar_root(text: str) -> tuple[int, str] | None:
    if not text or text[0] not in _WEIMAR_ROOTS:
        return None
    root = _WEIMAR_ROOTS[text[0]]
    index = 1
    while index < len(text) and text[index] in "#b":
        root += 1 if text[index] == "#" else -1
        index += 1
    return root % 12, text[index:]


@dataclass(frozen=True)
class BeatCell:
    """One beat of a Weimar solo's chord grid."""

    onset: float
    bar: int
    beat: int
    chord: JazzChord | None
    #: Raw symbol. An empty cell means "hold the previous chord" while "NC"
    #: means there is deliberately no chord; conflating them invents harmony.
    raw: str
    chorus: int


@dataclass
class Solo:
    """One Weimar solo: a real melody plus the changes played under it."""

    melid: int
    title: str
    performer: str
    tonic: int
    mode: str
    meter: tuple[int, int]
    tempo: float
    #: (onset seconds, pitch, duration seconds)
    notes: list[tuple[float, int, float]] = field(default_factory=list)
    cells: list[BeatCell] = field(default_factory=list)

    @property
    def beat_duration(self) -> float:
        return 60.0 / self.tempo if self.tempo else 0.5

    def chorus_range(self, chorus: int) -> tuple[int, int] | None:
        """Beat-index range [start, stop) of one chorus of the form."""
        indices = [i for i, cell in enumerate(self.cells) if cell.chorus == chorus]
        return (indices[0], indices[-1] + 1) if indices else None


def _parse_weimar_key(text: str) -> tuple[int, str] | None:
    if not text or "-" not in text:
        return None
    root_text, _, mode_text = text.partition("-")
    parsed = _weimar_root(root_text)
    if parsed is None:
        return None
    mode = "minor" if mode_text.startswith("min") else "major"
    return parsed[0], mode


def load_solos(
    *,
    path: Path = WJAZZD_PATH,
    download: bool = True,
    limit: int | None = None,
    meter: tuple[int, int] | None = (4, 4),
) -> list[Solo]:
    """Load solos with their beat-aligned chords and melody notes."""
    import sqlite3

    if download:
        _download(WJAZZD_URL, path)
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        solos: list[Solo] = []
        rows = db.execute(
            "SELECT melid, title, performer, key, signature, avgtempo FROM solo_info ORDER BY melid"
        ).fetchall()
        for melid, title, performer, key_text, signature, tempo in rows:
            key = _parse_weimar_key(key_text or "")
            if key is None or not signature or "/" not in signature:
                continue
            numerator, denominator = (int(x) for x in signature.split("/"))
            if meter is not None and (numerator, denominator) != meter:
                continue
            solo = Solo(
                melid=int(melid),
                title=title or "",
                performer=performer or "",
                tonic=key[0],
                mode=key[1],
                meter=(numerator, denominator),
                tempo=float(tempo or 0.0),
            )
            solo.notes = [
                (float(onset), int(pitch), float(duration))
                for onset, pitch, duration in db.execute(
                    "SELECT onset, pitch, duration FROM melody WHERE melid = ? ORDER BY onset", (melid,)
                )
            ]
            solo.cells = [
                BeatCell(
                    onset=float(onset),
                    bar=int(bar),
                    beat=int(beat),
                    chord=parse_weimar_chord(chord or ""),
                    raw=(chord or "").strip(),
                    chorus=int(chorus if chorus is not None else 0),
                )
                for onset, bar, beat, chord, chorus in db.execute(
                    "SELECT onset, bar, beat, chord, chorus_id FROM beats WHERE melid = ? ORDER BY onset",
                    (melid,),
                )
            ]
            if not solo.notes or not any(cell.chord for cell in solo.cells):
                continue
            solos.append(solo)
            if limit is not None and len(solos) >= limit:
                break
        return solos
    finally:
        db.close()


def solo_progression(solo: Solo) -> Progression:
    """The solo's own changes as a `Progression` in quarter-note beats.

    Weimar writes a chord symbol only where the harmony *changes*, leaving the
    cell empty while it is held. Treating an empty cell as silence would slice
    the changes into disconnected one-beat islands with holes between them, so
    each chord is held until the next symbol or an explicit "NC".
    """
    return _cells_to_progression(solo, solo.cells, origin=0.0)


def chorus_progression(solo: Solo, chorus: int = 1) -> Progression:
    """One chorus of the form, rebased to start at beat 0.

    A chorus is one pass through the tune, so this is the unit that compares
    like-for-like against a lead sheet: `chorus_progression(solo)` and the
    treebank entry for the same title are two performers' answers to the same
    question.
    """
    span = solo.chorus_range(chorus)
    if span is None:
        return Progression([], solo.tonic, solo.mode, solo.meter, solo.title, "wjazzd")
    start, stop = span
    return _cells_to_progression(solo, solo.cells[start:stop], origin=float(start))


def _cells_to_progression(solo: Solo, cells: Sequence[BeatCell], *, origin: float) -> Progression:
    spans: list[ChordSpan] = []
    held: JazzChord | None = None
    for index, cell in enumerate(cells):
        start = float(index)
        chord = cell.chord
        if chord is None and not cell.raw:
            chord = held  # empty cell: the previous chord is still sounding
        if chord is None:
            held = None
            continue
        held = chord
        if spans and spans[-1].chord.same_harmony(chord) and abs(spans[-1].stop - start) < 1e-6:
            spans[-1] = ChordSpan(spans[-1].start, spans[-1].duration + 1.0, spans[-1].chord)
        else:
            spans.append(ChordSpan(start, 1.0, chord))
    return Progression(
        spans=spans,
        tonic=solo.tonic,
        mode=solo.mode,
        meter=solo.meter,
        title=f"{solo.title} ({solo.performer})",
        source="wjazzd",
    )


def solo_melody_beats(solo: Solo) -> list[tuple[float, int, float]]:
    """Melody notes converted from seconds to quarter-note beats.

    Weimar timestamps are wall-clock seconds; the beat grid comes from the
    `beats` table, so the conversion is by interpolation between beat onsets
    rather than by dividing by an average tempo, which would drift over a
    three-minute solo.
    """
    onsets = [cell.onset for cell in solo.cells]
    if len(onsets) < 2:
        return []

    def to_beats(seconds: float) -> float:
        if seconds <= onsets[0]:
            step = onsets[1] - onsets[0]
            return (seconds - onsets[0]) / step if step > 0 else 0.0
        if seconds >= onsets[-1]:
            step = onsets[-1] - onsets[-2]
            return len(onsets) - 1 + (seconds - onsets[-1]) / step if step > 0 else float(len(onsets) - 1)
        low, high = 0, len(onsets) - 1
        while high - low > 1:
            mid = (low + high) // 2
            if onsets[mid] <= seconds:
                low = mid
            else:
                high = mid
        step = onsets[high] - onsets[low]
        return low + ((seconds - onsets[low]) / step if step > 0 else 0.0)

    out: list[tuple[float, int, float]] = []
    for onset, pitch, duration in solo.notes:
        start = to_beats(onset)
        stop = to_beats(onset + duration)
        out.append((start, pitch, max(0.125, stop - start)))
    return out


def chorus_melody(solo: Solo, chorus: int = 1) -> list[tuple[float, int, float]]:
    """Melody notes inside one chorus, rebased to start at beat 0."""
    span = solo.chorus_range(chorus)
    if span is None:
        return []
    start, stop = span
    return [
        (onset - start, pitch, duration)
        for onset, pitch, duration in solo_melody_beats(solo)
        if start - 1e-6 <= onset < stop
    ]
