# Jazz reharmonization: what was built, what was measured, and where the argument breaks

**Summary.** The strategic case for this workstream was that Bach chorale
harmonization has a right answer so search wins, while jazz reharmonization does
not so sampling should win. Half of that survived measurement, and it is the
half that matters for the product.

Sampling delivers genuine one-to-many variety at no measurable cost in quality:
five draws of the same tune differ in 27% of their chord roots, where the
deterministic engine differs in exactly 0%, and the headline score moves by 0.01
across a 30× change in temperature. That is the thing search structurally cannot
do, and it is real.

Sampling does **not** deliver adventure. Given an identical candidate space and
identical constraints, the argmax of hand-written substitution rules is *more*
chromatic than the learned sampler (0.112 vs 0.071 chromatic tones per chord
tone; the rules engine wins that on 20 tunes out of 20). A model of 1170
standards puts its mass where the corpus does, and the middle of the jazz corpus
is a diatonic ii-V-I. Adventurousness lives in the objective, not in the sampler
— which is why the shipped engine samples from a **hybrid** of the two.

On the one criterion no engine optimises for — distance from the changes a human
rhythm section actually played under the same melody — all three reharmonizers
beat the unreharmonized skeleton, and the learned ones are **not** systematically
better than the rules (mean paired difference 0.000 to 0.005 with a standard
deviation six times larger, 9/20 and 13/20 wins). That is a negative result and
it is stated as one.

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
   melodic weight over it", and the engines land at 0.031–0.042 aggregate —
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
| headline | 0.114 | **0.789** | 0.648 | 0.707 | 0.613 |
| hard melody conflicts | 0.200 | **0.031** | 0.042 | 0.040 | 0.061 |
| chord-tone rate | 0.539 | 0.603 | 0.570 | 0.558 | 0.591 |
| seventh rate | 0.079 | 0.935 | 0.976 | 0.982 | 0.833 |
| dominants that resolve | 0.237 | 0.845 | 0.611 | 0.774 | 0.725 |
| ii-V per 16 bars | 0.053 | 1.120 | 4.229 | 5.051 | 3.343 |
| chromatic tone rate | 0.049 | **0.112** | 0.071 | 0.085 | 0.141 |
| beats per chord | 2.458 | 2.367 | 2.452 | 2.435 | 4.895 |
| roots changed vs skeleton | 0.000 | 0.313 | 0.367 | 0.453 | 0.734 |
| **distance from human** | 0.628 | 0.583 | **0.579** | 0.583 | — |
| **sample diversity** | — | **0.000** | **0.266** | 0.230 | — |
| **style divergence** | 0.480 | 0.229 | 0.190 | **0.185** | 0.278 |

Seven traditional tunes, same configuration, no human reference:

| metric | skeleton | rules | sampled | hybrid |
| --- | ---: | ---: | ---: | ---: |
| headline | 0.295 | 0.865 | 0.761 | 0.835 |
| hard melody conflicts | 0.048 | 0.000 | 0.001 | 0.002 |
| seventh rate | 0.043 | 0.959 | 0.998 | 0.997 |
| ii-V per 16 bars | 0.254 | 1.024 | 3.303 | 4.039 |
| roots changed | 0.000 | 0.206 | 0.271 | 0.309 |
| sample diversity | — | 0.000 | 0.214 | 0.223 |
| style divergence | 0.445 | 0.331 | 0.312 | 0.312 |

### Paired per tune, because means hide whether a difference is systematic

| | hybrid − rules | | | sampled − rules | | |
| metric | mean | sd | wins | mean | sd | wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| headline | −0.082 | 0.033 | 0/20 | −0.140 | 0.041 | 0/20 |
| hard conflicts | +0.009 | 0.010 | 17/20 | +0.012 | 0.008 | 17/20 |
| chromatic tone rate | −0.027 | 0.020 | 1/20 | −0.041 | 0.020 | 0/20 |
| distance from human | −0.000 | 0.031 | 13/20 | −0.005 | 0.027 | 9/20 |
| style divergence | −0.044 | 0.018 | 0/20 | −0.039 | 0.018 | 0/20 |

Four of those five are near-unanimous in one direction or the other. Those are
real effects. "Distance from human" is a coin flip with a mean smaller than a
sixth of its own spread, and calling it a win for either side would be
dishonest.

---

## 4. Reading the table honestly

**The headline metric is contaminated and I do not lean on it.** Dominant
resolution is 25% of it, and the rules engine explicitly optimises dominant
resolution — it scores 0.845 there against the human reference's 0.725. An
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
  *more* typical than the human reference (0.185 vs 0.278), which is the same
  fact seen from the other side: a model of the corpus average writes the corpus
  average.
* *Distance from what the human played*: everything beats the skeleton
  (0.579–0.583 vs 0.628); nothing systematically beats anything else.
* *Diversity*: 0.266 versus exactly 0.000. This is not a marginal effect and it
  is the one search cannot have at any setting.

---

## 5. The dials

Both were calibrated by sweeping, not chosen by taste. 20 jazz tunes:

| temperature | headline | chromatic | diversity |
| ---: | ---: | ---: | ---: |
| 0.05 | 0.631 | 0.065 | 0.007 |
| 0.60 | 0.635 | 0.067 | 0.144 |
| 1.30 | 0.635 | 0.082 | 0.327 |
| 1.80 | 0.641 | 0.102 | 0.425 |

| adventure | rules: roots changed | rules: chromatic | hybrid: roots changed |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.014 | 0.063 | 0.267 |
| 0.50 | 0.147 | 0.090 | 0.382 |
| 1.00 | 0.439 | 0.146 | 0.491 |

Temperature buys variety at flat quality across a 30× range — that is what a
dial should do, and it is the product. `adventure` is the one that moves how far
out the harmony goes, and it is a hand-written cost, not a property of the
model.

---

## 6. Does it sound good?

Metrics cannot answer this, so here is the actual output. Shenandoah, whose
skeleton is `C/E | F | C | Bb/D | F | F | F | F | Bb | Dm | C | F | C`:

```
rules    Csus4(7) | Fmaj7 | C6 | Bbmaj7(9) | Fmaj7(13) | Dm7* | Fmaj7 | F7(13) |
         Bbmaj7#11 | G7* | C7(13) | Fmaj7 | C7
sampled  C7 | F6 | Gm7(11)* | C7(9)* | Fmaj7(13) | Fmaj7 | Cm7(11,13)* | F7(13) |
         Bbmaj7#11 | Gm7* | C7(13) | Fmaj7 | C7          (* = substitution)
```

The sampled version turns bars 3–4 into a ii-V (`Gm7 | C7`), turns bar 7 into
the related ii of the F7 that was already there so that `Cm7 F7 → Bbmaj7` is a
textbook ii-V into the subdominant, and makes that Bb a `maj7#11`. Those are
choices a player would make. Three different seeds produce three different but
equally defensible sets, which is the point.

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
chromatic tones 0.112 vs 0.071, unanimous across 20 tunes. If you want output
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
  bebop lines it starts at a 0.200 hard-conflict rate — worse than anything
  downstream produces. A jazz-native skeleton would probably improve the final
  result more than any further work on the substitution stage.
* **Harmonic rhythm is too fast**: 2.4 beats per chord against 4.9 in the
  reference (2.8 in the treebank). The reduction to bar and half-bar units
  inherits the rules engine's chord changes rather than deciding jazz phrasing
  for itself.
* **Chromaticism is still below the oracle** (0.112 best vs 0.141–0.184). Most
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
