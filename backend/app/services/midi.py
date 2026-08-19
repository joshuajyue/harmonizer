from __future__ import annotations

from collections import defaultdict
from io import BytesIO

import mido

from contracts.schema import (
    HarmonizeResponse,
    KeySignature,
    Melody,
    Note,
    TimeSignature,
    Voice,
)


class MidiConversionError(ValueError):
    pass


class MidiService:
    def __init__(self, max_upload_bytes: int) -> None:
        self._max_upload_bytes = max_upload_bytes

    def import_melody(self, data: bytes) -> Melody:
        if not data:
            raise MidiConversionError("The uploaded MIDI file is empty.")
        if len(data) > self._max_upload_bytes:
            raise MidiConversionError("The uploaded MIDI file is too large.")
        try:
            midi = mido.MidiFile(file=BytesIO(data), clip=True)
        except Exception as exc:
            raise MidiConversionError("The uploaded file is not valid MIDI.") from exc

        ticks_per_beat = midi.ticks_per_beat
        absolute_tick = 0
        first_tempo: int | None = None
        time_signature = TimeSignature(numerator=4, denominator=4)
        time_signature_seen = False
        key_signature: KeySignature | None = None
        active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        notes: list[Note] = []

        for message in mido.merge_tracks(midi.tracks):
            absolute_tick += message.time
            if message.type == "set_tempo" and first_tempo is None:
                first_tempo = message.tempo
            elif message.type == "time_signature" and not time_signature_seen:
                time_signature = TimeSignature(
                    numerator=message.numerator,
                    denominator=message.denominator,
                )
                time_signature_seen = True
            elif message.type == "key_signature" and key_signature is None:
                key_signature = _parse_midi_key(message.key)
            elif message.type == "note_on" and message.velocity > 0:
                active[(message.channel, message.note)].append(
                    (absolute_tick, message.velocity)
                )
            elif message.type in {"note_off", "note_on"}:
                key = (message.channel, message.note)
                if not active[key]:
                    continue
                start_tick, velocity = active[key].pop(0)
                notes.append(
                    Note(
                        pitch=message.note,
                        start=start_tick / ticks_per_beat,
                        duration=max(1, absolute_tick - start_tick) / ticks_per_beat,
                        velocity=velocity,
                    )
                )

        for (_, pitch), starts in active.items():
            for start_tick, velocity in starts:
                notes.append(
                    Note(
                        pitch=pitch,
                        start=start_tick / ticks_per_beat,
                        duration=max(1, absolute_tick - start_tick) / ticks_per_beat,
                        velocity=velocity,
                    )
                )

        notes.sort(key=lambda note: (note.start, note.pitch, note.duration))
        if first_tempo is None:
            raise MidiConversionError(
                "The MIDI file does not declare a tempo with a Set Tempo event."
            )
        return Melody(
            notes=notes,
            tempo=float(mido.tempo2bpm(first_tempo)),
            timeSignature=time_signature,
            key=key_signature,
        )

    def export_harmonization(
        self,
        harmonization: HarmonizeResponse,
        *,
        tempo: float,
    ) -> bytes:
        return self.voices_to_midi(
            harmonization.voices,
            tempo=tempo,
            key=harmonization.key,
        )

    def voices_to_midi(
        self,
        voices: list[Voice],
        *,
        tempo: float,
        key: KeySignature | None = None,
        time_signature: TimeSignature | None = None,
    ) -> bytes:
        ticks_per_beat = 480
        midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
        metadata = mido.MidiTrack()
        midi.tracks.append(metadata)
        metadata.append(mido.MetaMessage("track_name", name="HarmonAIzer v2", time=0))
        metadata.append(
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0)
        )
        signature = time_signature or TimeSignature(numerator=4, denominator=4)
        metadata.append(
            mido.MetaMessage(
                "time_signature",
                numerator=signature.numerator,
                denominator=signature.denominator,
                time=0,
            )
        )
        if key is not None:
            metadata.append(
                mido.MetaMessage("key_signature", key=_format_midi_key(key), time=0)
            )
        metadata.append(mido.MetaMessage("end_of_track", time=0))

        for index, voice in enumerate(voices):
            channel = index % 16
            if channel == 9:
                channel = 15
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.MetaMessage("track_name", name=voice.name.title(), time=0))
            track.append(mido.Message("program_change", program=52, channel=channel, time=0))

            events: list[tuple[int, int, mido.Message]] = []
            for note in voice.notes:
                start_tick = max(0, round(note.start * ticks_per_beat))
                end_tick = max(
                    start_tick + 1,
                    round((note.start + note.duration) * ticks_per_beat),
                )
                events.append(
                    (
                        start_tick,
                        1,
                        mido.Message(
                            "note_on",
                            note=note.pitch,
                            velocity=note.velocity,
                            channel=channel,
                            time=0,
                        ),
                    )
                )
                events.append(
                    (
                        end_tick,
                        0,
                        mido.Message(
                            "note_off",
                            note=note.pitch,
                            velocity=0,
                            channel=channel,
                            time=0,
                        ),
                    )
                )

            previous_tick = 0
            for tick, _, message in sorted(events, key=lambda event: (event[0], event[1])):
                message.time = tick - previous_tick
                previous_tick = tick
                track.append(message)
            track.append(mido.MetaMessage("end_of_track", time=0))

        output = BytesIO()
        midi.save(file=output)
        return output.getvalue()


def _parse_midi_key(value: str) -> KeySignature | None:
    minor = value.endswith("m")
    tonic_text = value[:-1] if minor else value
    if not tonic_text or tonic_text[0].upper() not in "ABCDEFG":
        return None
    pitch_classes = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    tonic = pitch_classes[tonic_text[0].upper()]
    for accidental in tonic_text[1:]:
        if accidental == "#":
            tonic += 1
        elif accidental == "b":
            tonic -= 1
        else:
            return None
    return KeySignature(tonic=tonic % 12, mode="minor" if minor else "major")


def _format_midi_key(key: KeySignature) -> str:
    major = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
    minor = ("Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm")
    return (minor if key.mode == "minor" else major)[key.tonic]
