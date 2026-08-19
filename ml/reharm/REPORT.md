# Jazz reharmonization: what was built, what was measured, and where the argument breaks

**Summary.** The strategic case for this workstream was that Bach chorale
harmonization has a right answer so search wins, while jazz reharmonization does
not so sampling should win. Half of that survived measurement, and it is the
half that matters for the product.

Sampling delivers genuine one-to-many variety at no measurable cost in quality:
five draws of the same tune differ in 22% of their chord roots, where the
deterministic engine differs in exactly 0%, and the headline score moves by 0.02
across a 36× change in temperature. That is the thing search structurally cannot
do, and it is real.

Sampling does **not** deliver adventure. Given an identical candidate space and
identical constraints, the argmax of hand-written substitution rules is *more*
chromatic than the learned sampler (0.109 vs 0.066 chromatic tones per chord
tone; the rules engine wins that on 20 tunes out of 20). A model of 1170
standards puts its mass where the corpus does, and the middle of the jazz corpus
is a diatonic ii-V-I. Adventurousness lives in the objective, not in the sampler
— which is why the shipped engine samples from a **hybrid** of the two.

On the one criterion no engine optimises for — distance from the changes a human
rhythm section actually played under the same melody — all three reharmonizers
beat the unreharmonized skeleton, and the learned ones are **not** systematically
better than the rules (mean paired difference 0.000 to 0.004 against a standard
deviation five to nine times larger, 10/20 and 11/20 wins). That is a negative
result and it is stated as one.

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
   melodic weight over it", and the engines land at 0.037–0.053 aggregate —
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
| headline | 0.108 | **0.777** | 0.612 | 0.678 | 0.613 |
| hard melody conflicts | 0.215 | **0.037** | 0.053 | 0.048 | 0.061 |
| chord-tone rate | 0.509 | 0.598 | 0.558 | 0.549 | 0.591 |
| seventh rate | 0.051 | 0.914 | 0.973 | 0.979 | 0.833 |
| dominants that resolve | 0.274 | 0.877 | 0.618 | 0.786 | 0.725 |
| ii-V per 16 bars | 0.053 | 1.143 | 3.129 | 3.663 | 3.343 |
| chromatic tone rate | 0.036 | **0.109** | 0.066 | 0.082 | 0.141 |
| beats per chord | 3.306 | 3.034 | 3.296 | 3.239 | 4.895 |
| roots changed vs skeleton | 0.000 | 0.319 | 0.337 | 0.423 | 0.733 |
| **distance from human** | 0.635 | 0.592 | 0.592 | **0.588** | — |
| **sample diversity** | — | **0.000** | **0.221** | 0.194 | — |
| **style divergence** | 0.476 | 0.242 | 0.216 | **0.209** | 0.278 |

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
| headline | −0.099 | 0.038 | 0/20 | −0.165 | 0.044 | 0/20 |
| hard conflicts | +0.011 | 0.014 | 16/20 | +0.017 | 0.016 | 16/20 |
| chromatic tone rate | −0.027 | 0.020 | 1/20 | −0.042 | 0.017 | 0/20 |
| distance from human | −0.004 | 0.036 | 10/20 | −0.000 | 0.020 | 11/20 |
| style divergence | −0.033 | 0.016 | 0/20 | −0.026 | 0.013 | 1/20 |

Four of those five are near-unanimous in one direction or the other. Those are
real effects. "Distance from human" is a coin flip with a mean smaller than a
sixth of its own spread, and calling it a win for either side would be
dishonest.

---

## 4. Reading the table honestly

**The headline metric is contaminated and I do not lean on it.** Dominant
resolution is 25% of it, and the rules engine explicitly optimises dominant
resolution — it scores 0.877 there against the human reference's 0.725. An
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
  (0.588–0.592 vs 0.635); nothing systematically beats anything else.
* *Diversity*: 0.221 versus exactly 0.000. This is not a marginal effect and it
  is the one search cannot have at any setting.

---

## 5. The dials

Both were calibrated by sweeping, not chosen by taste. 20 jazz tunes:

| temperature | headline | chromatic | diversity |
| ---: | ---: | ---: | ---: |
| 0.05 | 0.609 | 0.059 | 0.001 |
| 0.60 | 0.607 | 0.062 | 0.103 |
| 1.30 | 0.614 | 0.076 | 0.286 |
| 1.80 | 0.627 | 0.090 | 0.374 |

| adventure | rules: roots changed | rules: chromatic | hybrid: roots changed |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.103 | 0.062 | 0.277 |
| 0.50 | 0.184 | 0.079 | 0.375 |
| 1.00 | 0.431 | 0.148 | 0.473 |

Temperature buys variety at flat quality across a 36× range — that is what a
dial should do, and it is the product. `adventure` is the one that moves how far
out the harmony goes, and it is a hand-written cost, not a property of the
model.

---

## 6. Does it sound good?

Metrics cannot answer this, so here is the actual output. Shenandoah, whose
skeleton is `F | C | Bb/D | F | F | F | F | F | F | Bb | Dm | F | C`:

```
rules    Fmaj7 | C6 | Bbmaj7(9) | Fmaj7(13) | Dm7* | Fmaj7 | Dm7* | Fmaj7(13) |
         F7 | Bbmaj7#11 | Dm7 | Fmaj7 | C7
sampled  Fmaj7 | Gm7(11)* | C7(9)* | Fmaj7(13) | Fmaj7 | Fmaj7 | Fmaj7 |
         Fmaj7(13) | F7 | Bbmaj7#11 | C7(9)* | Fmaj7 | C7      (* = substitution)
```

The sampled version turns bars 2–3 into a ii-V (`Gm7 | C7`) instead of the plain
`C | Bb` the skeleton had, keeps `F7 → Bbmaj7#11` as a dominant into the
subdominant with a lydian colour on top, and turns the penultimate bar into the
dominant that sets up the last one. Those are choices a player would make.
Different seeds produce different but equally defensible sets, which is the
point; the tritone substitutions and borrowed chords show up on tunes with more
chromatic room in the melody, such as `--tune greensleeves --temperature 1.3`,
where a `Bb7#11` arrives as the tritone substitute of the V and resolves down a
semitone into the tonic minor.

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
chromatic tones 0.109 vs 0.066, unanimous across 20 tunes. If you want output
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

## 8. Limitations

* **The skeleton is the weakest link.** It comes from a chorale engine, and on
  bebop lines it starts at a 0.215 hard-conflict rate — worse than anything
  downstream produces. A jazz-native skeleton would probably improve the final
  result more than any further work on the substitution stage.
* **Harmonic rhythm is decided by the melody, not by phrasing**: 3.0–3.3 beats
  per chord, which sits between the treebank's 2.8 and the reference's 4.9, but
  it comes from where the melody articulates the middle of the bar rather than
  from any notion of where a phrase wants a chord change.
* **Chromaticism is still below the oracle** (0.109 best vs 0.141–0.184). Most
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
