# `ml/reharm` — jazz reharmonization

Takes a melody, harmonizes it with the existing functional rules engine to get a
clean diatonic skeleton, then **reharmonizes** that skeleton with the jazz
substitution vocabulary — tritone substitutions, backdoor and secondary
dominants, modal interchange, passing diminished chords, chromatic approach,
relative swaps, ii-V insertions, optional Coltrane changes — and voices the
result with jazz voicings rather than chorale ones.

Two engines are registered and both appear in `GET /api/v1/engines`:

| id | learned | what it does |
| --- | --- | --- |
| `jazz_reharm` | yes | Learned chord model + substitution vocabulary, **sampled** at a temperature. Different every seed. |
| `jazz_reharm_rules` | no | The same vocabulary and constraints, chosen by **Viterbi argmax** over hand-written functional scores. Deterministic. |

They share the candidate space, the hard constraints and the voicer, and differ
only in how a path is chosen. That is what makes the comparison in
[`REPORT.md`](REPORT.md) mean anything.

## Why this exists

The rules engine is diatonic-functional *by construction*: it cannot choose to
be surprising, and it emits triads (measured: 7.9% of its chords are sevenths or
sixths, against 93% in real jazz lead sheets). Reharmonization is where a
learned or stochastic model has a structural advantage, because unlike chorale
harmonization it is a genuine one-to-many mapping — there is no correct tritone
substitution, and the argmax of a reharmonization distribution is by definition
the safe one.

How much of that argument survived measurement is the subject of `REPORT.md`.
Short version: sampling buys real variety and costs nothing, but it does not buy
adventure — that comes from the objective, not from the sampler.

## Try it

```bash
# see and hear a reharmonization (writes two MIDI files)
python -m ml.reharm.demo --tune shenandoah --seeds 3 --midi out.mid
python -m ml.reharm.demo --tune jazz:0 --temperature 1.4

# what real jazz measures, by our own metrics — run this before believing anything
python -m ml.reharm.oracle

# train the chord model (already committed; this reproduces it)
python -m ml.reharm.model

# the comparison
python -m ml.reharm.evaluate --cases jazz --limit 20 --samples 5
python -m ml.reharm.evaluate --cases traditional --sweep temperature
```

From the API, nothing new is needed — it is the existing `HarmonyEngine`
interface:

```jsonc
POST /api/v1/harmonize
{ "melody": {...}, "engine": "jazz_reharm", "options": { "temperature": 1.0, "seed": 7 } }
```

`temperature: 0` is argmax and exactly reproducible, per the contract. It is
also the least interesting setting; **1.0 is the recommended default**, and
`seed` selects between equally valid reharmonizations.

## The dials

Two, and both were calibrated by measurement rather than taste (numbers from a
20-tune jazz set, `--sweep`):

| dial | 0.0 | 1.0+ | what it moves |
| --- | --- | --- | --- |
| `adventure` | roots changed 0.01 | roots changed 0.44 | how far the harmony travels from the tune. Scales the substitution cost and the anchor. |
| `temperature` | diversity 0.007 | diversity 0.425 | how different two runs are. Barely touches quality (headline 0.631 → 0.641). |

`adventure` is a `ReharmConfig` field; `temperature` is the contract's own
option, so the UI already has it.

## Architecture

```
melody
  └─ skeleton.py      run ml/engines/rules.py, reduce to bar / half-bar units
      └─ substitutions.py   generate candidates per unit, melody-safe by construction
          └─ search.py      Viterbi (rules) or forward-filtering backward-sampling (learned)
              └─ voicing.py rootless / quartal / upper-structure voicings + bass
                  └─ engine.py  Harmonization: voices, chords with provenance, violations
```

* **`chords.py`** — `JazzChord` (core quality + explicit extensions, mirroring
  `contracts.schema.Chord`), chord-symbol parsing, and the avoid-note model.
* **`data.py`** — the two corpora, downloaded to a gitignored cache.
* **`metrics.py`** — jazz metrics. Explicitly *not* the chorale harness.
* **`oracle.py`** — real jazz scored by those metrics. Run it first.
* **`model.py`** — interpolated trigram over key-relative chord tokens.
* **`evaluate.py`** — the four-way comparison plus paired per-tune statistics.
* **`demo.py`** — printed changes and a MIDI file, with a dependency-free writer.

### Sampling

Sampling is forward-filtering backward-sampling over the whole lattice, not
greedy left-to-right choice. A greedy sampler picks a tritone substitute in bar
3 and finds out in bar 4 that nothing can follow it. FFBS computes exact
backward messages first, so every draw is a sample from the whole-sequence
distribution and is coherent by construction — and as `T → 0` it provably
converges on the Viterbi path, which is tested.

### Melody compatibility is a hard constraint

Every candidate chord is checked against the melody sounding over it *during
generation*, so the lattice contains only chords the melody can live over and
the sampler never has to reject-sample. The model is one rule with named
exceptions: a melody note a semitone above a voiced chord tone is a conflict,
except that b9/#9/#11/b13 are available tensions on a dominant, and a clash
against the fifth or against the third of a dominant is repaired by dropping
that note (which is what makes 7sus4 chords exist).

Where the melody lands on a tension, the chord *states* it — a b9 in the tune
becomes a stated b9 in the chord rather than an accident the voicing dodges.

### Provenance

Every substituted chord carries `substitutionOf` (the roman numeral the rules
engine produced, verbatim) and `substitutionKind`. The UI can therefore say
"this bII7 replaced your V7, via tritone substitution" instead of just emitting
it. `demo.py` prints the same thing.

One wrinkle worth knowing: the related **ii** of an inserted secondary dominant
is tagged `secondary_dominant`, because the contract's `substitutionKind`
enumeration has no `related_ii`. Adding one would be a small contract change and
would read better in the UI.

## Data and licences

Downloaded on first use to `ml/reharm/cache/` (gitignored, never committed).

* **Jazz Harmony Treebank** — Harasim, Finkensiep, Ericson, O'Donnell &
  Rohrmeier, *The Jazz Harmony Treebank*, ISMIR 2020.
  <https://github.com/DCMLab/JazzHarmonyTreebank>. **CC BY 4.0.**
  1170 chord sequences, 150 with hierarchical analyses; 59,150 chords, all of
  which parse.
* **Weimar Jazz Database v2.1** — The Jazzomat Research Project, 2012–2017.
  <https://jazzomat.hfm-weimar.de>. **ODbL 1.0**, contents under **DbCL 1.0**.
  456 transcribed solos with beat-aligned changes; 29,443 chord cells, all of
  which parse.

Raw iRealPro data is **not** openly licensed and is deliberately not used; the
treebank's derived CC BY 4.0 release is. Test melodies are traditional tunes
long out of copyright, hand-encoded in `melodies.py`.

The only committed artefact is `assets/jazz_ngram.json` (272 KB), the trained
chord model, so the engine works from a fresh clone with no corpus.

## Tests

```bash
pytest ml/reharm            # 129 tests, ~10s
```

Corpus tests skip themselves when the cache is absent, so CI passes without a
46 MB download. The registered engines also pass the shared engine contract in
`ml/tests/test_engines.py` unchanged.

## Backend discovery

Nothing to do: `backend/app/services/engines.py` scans `ml.engines` **and**
`ml.reharm` (`DEFAULT_ENGINE_PACKAGES`), so importing this package registers
both engines and they appear in `GET /api/v1/engines` on their own. Verified
end to end through the real app:

```
GET  /api/v1/engines   -> fixed_thirds, jazz_reharm, jazz_reharm_rules, neural, ... , rules
POST /api/v1/harmonize -> engine "jazz_reharm", 67 ms, chords carry substitutionOf/substitutionKind
```
