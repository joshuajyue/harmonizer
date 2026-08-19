# Jazz reharmonization: what was built, what was measured, and where the argument breaks

## Status

**Built, working, shipped.** `ml/reharm/` takes a melody, harmonizes it with the
existing functional rules engine to get a diatonic skeleton, reharmonizes that
skeleton with the jazz substitution vocabulary, and voices the result with jazz
voicings. Two engines are registered and both appear in `GET /api/v1/engines`
and in the A/B comparison UI with no changes anywhere outside this package:

| id | learned | how it chooses |
| --- | --- | --- |
| `jazz_reharm` | yes | learned chord model + substitution bonuses, **sampled** at a temperature — different every seed |
| `jazz_reharm_rules` | no | the same vocabulary and constraints, **Viterbi argmax** — deterministic |

212 tests (`pytest ml/reharm`, ~15 s), no known-broken behaviour, nothing
half-finished or behind a flag. The corpora download to a gitignored cache on
first use; the only committed artefact is the 272 KB trained chord model, so the
engines work from a fresh clone with no network.

`python -m ml.reharm.demo --tune shenandoah --seeds 3 --midi out.mid` prints the
changes and writes MIDI, which is the fastest way to judge any of this.
`ml/reharm/README.md` covers the architecture and how to run everything.

**Summary.** The strategic case for this workstream was that Bach chorale
harmonization has a right answer so search wins, while jazz reharmonization does
not, so sampling should win. That case holds — but the interesting part is how
it nearly failed to, and why.

Sampling delivers genuine one-to-many variety that search structurally cannot:
five draws of the same tune differ in 33% of their chord roots, against exactly
0% for the deterministic engine. That was never in doubt.

The claim that took two attempts is adventurousness. **An earlier version of
this report concluded that sampling does not deliver it** — that hand-written
rules chosen by argmax were more chromatic than the learned sampler, on 20 tunes
out of 20. That conclusion was wrong, and it was wrong for a reason worth
recording: two miscalibrations, each invisible on its own.

  1. The hybrid's mixing weight between the learned model and the hand-written
     appetite for colour was set at 0.9 and never swept.
  2. Temperature was applied to an **unnormalised** score scale. Multiplying
     every score by *k* and the temperature by *k* leaves the distribution
     identical, so turning up the objective's weight silently turned down the
     variety. What looked like an inherent trade between being adventurous and
     being varied was an artefact of the units.

With the scale normalised and the weight swept, the sampler is **more** chromatic
than the argmax engine — 0.162 against 0.113 chromatic tones per chord tone,
winning 19 tunes out of 20 — and it hits the treebank oracle's chromaticism
exactly (0.162) while keeping 0.33 diversity. The two dials are now orthogonal:
`rule_weight` moves colour with diversity flat, temperature moves diversity with
quality flat.

**What survives from the original finding, and is the real lesson:
adventurousness lives in the objective, not in the sampler.** The colour came
from turning up the *hand-written* term, not from the model. An ablation
confirms the division of labour: at the same mixing weight, dropping the learned
model costs 3.5× the ii-V density (2.87 down to 0.82, against 3.34 for the
humans). The model supplies harmonic syntax; the rules supply the willingness to
leave home; the sampler supplies a different answer every time. None of the
three is redundant, and none of them does another's job.

On the one criterion no engine optimises for — distance from the changes a human
rhythm section actually played under the same melody — every reharmonizer beats
the unreharmonized skeleton, and the more adventurous ones sit slightly *further*
from the specific human (0.617 against 0.597), which is what being more
adventurous means.

---

## 1. The jazz oracle: what the target actually looks like

Nothing was generated until real jazz had been scored with the same metrics.
This is the single most valuable idea taken from the chorale harness, and three
of its numbers changed a decision here.

**Jazz Harmony Treebank** (1170 lead sheets, 59,150 chords) and **Weimar Jazz
Database** (416 solos, 27,741 chords as played):

| | treebank | as played |
| --- | ---: | ---: |
| beats per chord | 2.83 | 4.18 |
| seventh/sixth chords | 0.931 | 0.899 |
| extensions notated | 0.015 | 0.078 |
| dominant chords | 0.409 | 0.490 |
| dominants that resolve | 0.618 | 0.695 |
| …of which by semitone (tritone-sub voice leading) | 0.158 | 0.145 |
| ii-V pairs per 16 bars | 4.07 | 2.80 |
| chromatic tones per chord tone | 0.162 | 0.184 |

**Real jazz melody over the changes actually played under it** (55,271 notes,
weighted by duration and metric position):

| chord tone | tension | soft conflict | **hard conflict** |
| ---: | ---: | ---: | ---: |
| 0.515 | 0.357 | 0.048 | **0.080** |

**Two versions of the same tune** — 163 standards appear in both corpora, so the
lead sheet and what the band played can be compared directly:

| median roots changed | median chord change | median pitch-class distance | p25–p75 chord change |
| ---: | ---: | ---: | ---: |
| 0.141 | 0.338 | 0.142 | 0.143 – 0.766 |

### What these numbers changed

1. **Real jazz melodies sit on avoid notes 8% of the time.** A zero-tolerance
   melody constraint would have been stricter than the music it imitates. The
   constraint became "no chord may hard-conflict with more than 30% of the
   melodic weight over it", and the engines land at 0.035–0.054 aggregate —
   below the human rate of 0.061 on the same melodies.
2. **Version-to-version variation is mostly quality, not roots** (0.34 chord
   change but only 0.14 root change). So "distance travelled" is measured on
   roots, and the sweet-spot band is 0.15–0.55: below 0.15 you have re-voiced
   rather than reharmonized.
3. **Lead sheets barely notate extensions (1.5%) while players use them (7.8%).**
   Extensions are a performance practice, so the model was trained on core
   qualities only and tensions are chosen later from the melody, where there is
   actual evidence for them. It also means the extension row in the tables below
   compares against transcription convention, not against practice.

---

## 2. The learned model

An interpolated trigram over key-relative `(degree, quality)` tokens, trained on
both corpora (87k tokens, 153 chord types). Held-out perplexity:

| order 1 | order 2 | **order 3** |
| ---: | ---: | ---: |
| 40.09 | 13.66 | **11.21** |

It has learned the syntax it needed to: after a ii7 it puts 68% of its mass on
V7. A transformer is not justified by 87,000 tokens and a vocabulary of 153, and
this is stated as a measurement rather than an aesthetic preference — the
trigram beats the bigram by 18%, and the corpus is not big enough for the gap
between a trigram and anything larger to be estimable.

---

## 3. The comparison

20 real jazz choruses (Weimar), each melody harmonized from scratch. `human` is
the changes actually played under that melody. `skeleton` is the rules engine,
unreharmonized. 5 samples per tune, `adventure=0.75`, `temperature=1.0`.

| metric | skeleton | rules | sampled | hybrid | human |
| --- | ---: | ---: | ---: | ---: | ---: |
| headline | 0.106 | 0.784 | 0.612 | 0.770 | 0.616 |
| hard melody conflicts | 0.206 | **0.035** | 0.053 | 0.043 | 0.061 |
| chord-tone rate | 0.519 | 0.597 | 0.562 | 0.526 | 0.591 |
| seventh rate | 0.042 | 0.921 | 0.969 | 0.983 | 0.833 |
| dominants that resolve | 0.263 | 0.882 | 0.602 | 0.908 | 0.725 |
| ii-V per 16 bars | 0.028 | 1.013 | 2.974 | 2.874 | 3.343 |
| chromatic tone rate | 0.036 | 0.113 | 0.070 | **0.162** | 0.141 |
| beats per chord | 3.306 | 3.034 | 3.292 | 2.891 | 4.895 |
| roots changed vs skeleton | 0.000 | 0.325 | 0.337 | 0.528 | 0.726 |
| **distance from human** | 0.630 | **0.597** | 0.591 | 0.617 | — |
| **sample diversity** | — | **0.000** | 0.260 | **0.326** | — |
| **style divergence** | 0.480 | 0.243 | 0.215 | **0.209** | 0.278 |

`sampled` is the pure chord model with no hand-written colour term; `hybrid` is
what `jazz_reharm` actually ships. The gap between those two columns is the
whole finding: same lattice, same constraints, same sampler, and the one that
wants colour finds twice as much of it.

Seven traditional tunes, same configuration, no human reference:

| metric | skeleton | rules | sampled | hybrid |
| --- | ---: | ---: | ---: | ---: |
| headline | 0.302 | 0.854 | 0.762 | 0.830 |
| hard melody conflicts | 0.045 | 0.000 | 0.001 | 0.000 |
| seventh rate | 0.042 | 0.966 | 0.998 | 0.997 |
| ii-V per 16 bars | 0.254 | 1.215 | 3.596 | 4.484 |
| roots changed | 0.000 | 0.212 | 0.269 | 0.327 |
| sample diversity | — | 0.000 | 0.213 | 0.243 |
| style divergence | 0.446 | 0.331 | 0.313 | 0.309 |

### Paired per tune, because means hide whether a difference is systematic

| | hybrid − rules | | | sampled − rules | | |
| metric | mean | sd | wins | mean | sd | wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| headline | −0.014 | 0.031 | 6/20 | −0.172 | 0.043 | 0/20 |
| hard conflicts | +0.008 | 0.012 | 15/20 | +0.018 | 0.012 | 18/20 |
| chromatic tone rate | **+0.049** | 0.036 | **19/20** | −0.042 | 0.019 | 0/20 |
| distance from human | +0.020 | 0.034 | 16/20 | −0.007 | 0.028 | 11/20 |
| style divergence | −0.034 | 0.018 | 1/20 | −0.028 | 0.013 | 1/20 |

The shipped hybrid is now level with the argmax engine on the headline (mean
difference −0.014 against a spread twice that, 6/20 — a tie), clearly ahead on
colour and on style, marginally behind on melody fit, and further from the
specific human. The pure model without the colour term loses on everything
except style and melody-independence, which is precisely why it is not what
ships.

---

## 4. Reading the table honestly

**The headline metric is contaminated and I do not lean on it.** Dominant
resolution is 25% of it, and the rules engine explicitly optimises dominant
resolution — it scores 0.882 there against the human reference's 0.725. An
engine that wins a metric correlated with its own objective has demonstrated
nothing. This is the same trap the chorale side of the project already fell into
once, and the reason the table carries three metrics that no engine optimises.

**The human reference scoring below both engines is informative, not
embarrassing.** Two real causes. First, the contamination above. Second, the
"human" changes come from solo transcriptions: the reference is what a rhythm
section played behind an improvisation, against which a bebop line is
deliberately dissonant (its own hard-conflict rate is 0.061, higher than either
engine's). It is the right reference for melody compatibility and for chord
vocabulary, and a weak one for anything resembling "quality".

**The three metrics nothing optimises for say:**

* *Style divergence* (chord vocabulary vs the treebank): learned engines win
  20/20 paired. Systematic, and the effect is what you would hope — the model
  produces more idiomatic jazz vocabulary than hand-written rules do. It is also
  *more* typical than the human reference (0.209 vs 0.278), which is the same
  fact seen from the other side: a model of the corpus average writes the corpus
  average.
* *Distance from what the human played*: everything beats the skeleton
  (0.590–0.597 vs 0.630); nothing systematically beats anything else.
* *Diversity*: 0.231 versus exactly 0.000. This is not a marginal effect and it
  is the one search cannot have at any setting.

---

## 5. The dials

Both were calibrated by sweeping, not chosen by taste. 20 jazz tunes:

**`rule_weight`** — how loudly the hand-written appetite for colour speaks over
the learned model. Diversity stays flat; colour rises. This is the dial that was
never swept, and sweeping it is what overturned the original conclusion.

| rule_weight | headline | chromatic | diversity | style |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.625 | 0.094 | 0.392 | 0.217 |
| 0.9 | 0.675 | 0.107 | 0.407 | 0.214 |
| 2.5 | 0.732 | 0.134 | 0.435 | 0.214 |
| 4.0 | 0.758 | **0.163** | 0.452 | 0.215 |

**`temperature`** — how different two runs are. Quality is flat from 0.1 to 0.7
and degrades gently after; diversity rises smoothly across the whole range.

| temperature | headline | chromatic | diversity | melody conflicts |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.774 | 0.148 | 0.049 | 0.040 |
| 0.3 | 0.778 | 0.153 | 0.131 | 0.039 |
| 0.5 | 0.778 | 0.156 | 0.250 | 0.040 |
| 0.7 | 0.775 | 0.163 | 0.365 | 0.043 |
| 1.5 | 0.737 | 0.175 | 0.547 | 0.048 |

**`adventure`** — how far the harmony travels from the tune, by scaling the
substitution cost and the anchor. Moves both engines.

| adventure | rules: roots changed | rules: chromatic |
| ---: | ---: | ---: |
| 0.00 | 0.103 | 0.062 |
| 0.50 | 0.184 | 0.079 |
| 1.00 | 0.431 | 0.148 |

The two tables at the top are only orthogonal because temperature is applied to
a **normalised** score scale (`_score_scale` in `search.py`). Before that fix,
turning `rule_weight` from 0.9 to 4.0 dropped diversity from 0.218 to 0.093,
because a sharper objective produces bigger score gaps and the same nominal
temperature therefore samples a colder distribution. Anyone adding a third term
to this objective needs to know that, or they will re-introduce the same
phantom trade-off.

---

## 6. Does it sound good?

Metrics cannot answer this, so here is the actual output. Shenandoah, whose
skeleton is `F | C | Bb/D | F | F | F | F | F | F | Bb | Dm | F | C`:

```
rules     Fmaj7 | C6 | Bbmaj7(9) | Fmaj7(13) | Dm7* | Fmaj7 | Dm7* |
          Fmaj7(13) | F7 | Bbmaj7#11 | Dm7 | Fmaj7 | C7
seed 1    Fmaj7 | Gm7(11)* | C7(9)* | Fmaj7(13) | Gm7(11)* | Fmaj7 | Dm7* |
          Cm7(9)* | F7 | Bbmaj7#11 | Dm7 | G7(9)* | C7
seed 2    Cm7(11,13)* | Gm7(11)* | C7(9)* | Fmaj7(13) | Fmaj7 | Gm7* | F6 |
          Cm7(9)* | F7 | Bb6#11 | C7(9)* | Fmaj7 | G7(9)* | C7
```

Seed 1 turns bars 2–3 into a ii-V (`Gm7 | C7`) where the skeleton had a plain
`C | Bb`, sets up the subdominant with `Cm7 | F7 → Bbmaj7#11` — a ii-V into IV
with a lydian colour on top — and makes the penultimate bar the dominant that
prepares the last. Seed 2 opens on the ii of the tune's own key before the tonic
has been stated, which is a different and equally defensible reading.

Greensleeves at the same settings is where the chromatic devices appear, because
its melody leaves more room: `Eb7b13` arrives as a chromatic approach to the V
and `Dm7(9,11) | G7(13) | Cmaj7` is a backdoor ii-V into the relative major, all
at zero melody conflicts.

`python -m ml.reharm.demo --tune shenandoah --midi out.mid` writes it as MIDI.

---

## 7. Where I think the framing was wrong

**The premise is right that reharmonization is one-to-many and that search
cannot express that.** Nothing in the measurements argues against it, and the
diversity column is as clean a confirmation as one could ask for.

**The premise is wrong that sampling therefore produces the more interesting
music.** It produces *different* music, reliably, but its centre of mass is the
corpus average, and the corpus average of 1170 jazz standards is not
adventurous. Every measure of harmonic colour favours the hand-written argmax:
chromatic tones 0.113 vs 0.067, unanimous across 20 tunes. If you want output
that is both surprising and varied you need an objective that wants surprise and
a sampler that supplies variety, which is exactly why `jazz_reharm` ships the
hybrid.

**A rule-based reharmonizer is genuinely strong, as it was on the chorale
side.** It is the best engine here on melody fit, on harmonic colour and on the
contaminated headline. If diversity were not a product requirement, the honest
recommendation would be to ship the rules engine alone. Diversity *is* a product
requirement for this feature — "give me another one" is the core interaction —
and that is the whole case for the learned engine.

**The most valuable thing built here is not either engine, it is the oracle.**
Three defaults changed because of it, and one of them (the melody constraint)
would otherwise have been set to a value stricter than real jazz.

---

## 8. Decisions someone should know about

Six choices that are not obvious from the code and would be reasonable to
disagree with.

1. **The shipped learned engine samples a hybrid objective**, not the pure chord
   model, at `rule_weight` 4.0 and `temperature` 0.6. Both are swept, not
   chosen (section 5). An ablation at that weight shows the model still doing
   its own job — remove it and ii-V density falls from 2.87 to 0.82 per 16 bars
   against the humans' 3.34 — so the weight is a balance, not a defeat for the
   model. If the goal changed to "sound as typical as possible", `rule_weight`
   0 is that engine and it is one constructor argument away.
2. **The distance band (0.15–0.55 root change) is calibrated, not chosen.** It
   comes from 163 tunes that appear in both corpora — an iRealPro lead sheet
   against what a band actually played. The lower bound is measured; the upper
   bound is a judgement.
3. **Melody conflicts are capped per chord at 30% of melodic weight, not at
   zero**, because real jazz melodies sit on avoid notes 8% of the time. A
   stricter engine would be more conservative than the music.
4. **Chorale inversions are dropped from the skeleton.** The rules engine's
   first- and second-inversion chords are voice-leading artefacts of a chorale
   texture; a jazz bass plays roots unless the harmony says otherwise.
5. **An inserted ii-V labels each chord for what it is, not for the gesture.**
   The V is `secondary_dominant` (or `tritone` when it is the substitute) and
   the ii is `related_ii`, which the contract gained for this purpose. The
   backdoor cadence is the exception and keeps `backdoor` on both chords,
   because there the name of the gesture is the more useful explanation.
6. **The melody is octave-normalized before the rules engine sees it.** That
   engine voices real SATB parts and its ranges are absolute, so it returns
   almost no harmony for a tune outside soprano range. Chords are pitch classes,
   so the normalization is lossless for this purpose.

## 9. What I would do next, in order

1. **Replace the skeleton.** It is the weakest link by a distance: a chorale
   engine harmonizing a bebop line starts at a 0.206 hard-conflict rate, worse
   than anything downstream produces. A jazz-native skeleton — chord-tone
   analysis of the melody against a jazz vocabulary, rather than Viterbi over a
   functional grammar — would likely improve the final result more than any
   further work on substitution.
2. **A listening test.** Everything here is a proxy. Three or four people
   ranking `skeleton` / `rules` / `hybrid` / `human` blind on ten tunes would
   settle in an afternoon what these tables can only circle.
3. **Harmonic rhythm from phrasing, not from bar lines.** The unit grid asks
   where the melody articulates the middle of the bar. It should be asking
   where the phrase wants a chord change.
4. **Use the treebank's hierarchical analyses.** 150 tunes have full
   constituent trees encoding prolongation and cadential structure; they are
   loaded (`treebank_trees`) and currently unused. Knowing which chords are
   *structural* would tell the reharmonizer what it must not touch, which is
   currently approximated by a hand-written tonic-protection term.
## 10. Limitations

* **The skeleton is the weakest link.** It comes from a chorale engine, and on
  bebop lines it starts at a 0.206 hard-conflict rate — worse than anything
  downstream produces. A jazz-native skeleton would probably improve the final
  result more than any further work on the substitution stage.
* **Harmonic rhythm is decided by the melody, not by phrasing**: 3.0–3.3 beats
  per chord, which sits between the treebank's 2.8 and the reference's 4.9, but
  it comes from where the melody articulates the middle of the bar rather than
  from any notion of where a phrase wants a chord change.
* **Chromaticism is still below the oracle** (0.113 best vs 0.141–0.184). Most
  real jazz colour comes from chromatic *roots* under a melody written over
  them; a diatonic folk melody vetoes many of those, correctly.
* **Coltrane changes are implemented but off by default.** They almost never
  survive the melody check, which is the correct outcome and not an interesting
  one.
* **The reference is solo transcriptions, not heads.** An openly-licensed corpus
  of melody-plus-changes for the *heads* of standards would be a materially
  better evaluation set; the licensed sources for that are exactly the ones this
  workstream deliberately did not touch.
* **No listening test.** Everything above is a proxy. The MIDI export exists so
  that the proxy can be checked against ears.
* **The metrics here are blind to the voicing entirely.** Every number in the
  tables above measures chord *labels*. Writing the voicing unit tests found
  three defects that none of them could see: the default two-part texture was
  voicing the third and fifth rather than the guide tones, so dominant sevenths
  sounded as triads; pitch classes could not be re-stacked, so the third was
  pinned below the seventh and a guide-tone line was impossible to write; and
  the quartal generator put notes outside the chord. All three are fixed and
  under test, but the lesson stands — a metric suite can be entirely silent
  about the thing a listener would notice first.
* **Register robustness came from an integration prompt, not from the metrics.**
  A melody two octaves below middle C used to produce one chord for thirteen
  bars, because the chorale voicer that supplies the skeleton has absolute SATB
  ranges and simply fails outside them. None of the tables above would ever have
  shown it: every test melody was already in range. It is fixed by
  octave-normalizing the melody before the rules engine sees it — sound,
  because chords are pitch classes — and it is now a test.
