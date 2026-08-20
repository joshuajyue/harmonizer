from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from heapq import heappop, heappush
from io import BytesIO
from math import isfinite

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
    def __init__(self, max_upload_bytes: int, max_notes: int = 10_000) -> None:
        self._max_upload_bytes = max_upload_bytes
        self._max_notes = max_notes

    def import_melody(self, data: bytes) -> Melody:
        if not data:
            raise MidiConversionError("The uploaded MIDI file is empty.")
        if len(data) > self._max_upload_bytes:
            raise MidiConversionError("The uploaded MIDI file is too large.")
        try:
            _preflight_midi(data, max_notes=self._max_notes)
            midi = mido.MidiFile(file=BytesIO(data), clip=True)
            return self._convert_midi(midi)
        except MidiConversionError:
            raise
        except Exception as exc:
            raise MidiConversionError("The uploaded file is not valid MIDI.") from exc

    def _convert_midi(self, midi: mido.MidiFile) -> Melody:
        ticks_per_beat = midi.ticks_per_beat
        if ticks_per_beat <= 0:
            raise MidiConversionError(
                "SMPTE or zero MIDI time division is unsupported; "
                "use a positive ticks-per-beat value."
            )
        first_tempo: int | None = None
        time_signature = TimeSignature(numerator=4, denominator=4)
        time_signature_seen = False
        key_signature: KeySignature | None = None
        active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        notes: list[Note] = []
        started_notes = 0
        final_tick = 0

        for absolute_tick, message in _iter_merged_messages(midi.tracks):
            final_tick = max(final_tick, absolute_tick)
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
                started_notes += 1
                if started_notes > self._max_notes:
                    raise MidiConversionError(
                        f"MIDI contains more than {self._max_notes} notes."
                    )
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
                        duration=max(1, final_tick - start_tick) / ticks_per_beat,
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
        time_signature: TimeSignature,
    ) -> bytes:
        return self.voices_to_midi(
            harmonization.voices,
            tempo=tempo,
            key=harmonization.key,
            time_signature=time_signature,
        )

    def voices_to_midi(
        self,
        voices: list[Voice],
        *,
        tempo: float,
        key: KeySignature | None = None,
        time_signature: TimeSignature,
    ) -> bytes:
        _validate_export_parameters(
            tempo=tempo,
            time_signature=time_signature,
        )
        note_count = sum(len(voice.notes) for voice in voices)
        if note_count > self._max_notes:
            raise MidiConversionError(
                f"MIDI export contains {note_count} notes; the limit is {self._max_notes}."
            )
        ticks_per_beat = 480
        midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
        metadata = mido.MidiTrack()
        midi.tracks.append(metadata)
        metadata.append(mido.MetaMessage("track_name", name="HarmonAIzer v2", time=0))
        metadata.append(
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0)
        )
        metadata.append(
            mido.MetaMessage(
                "time_signature",
                numerator=time_signature.numerator,
                denominator=time_signature.denominator,
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


def _iter_merged_messages(
    tracks: list[mido.MidiTrack],
) -> Iterator[tuple[int, mido.Message | mido.MetaMessage]]:
    heap: list[
        tuple[
            int,
            int,
            int,
            mido.Message | mido.MetaMessage,
            Iterator[mido.Message | mido.MetaMessage],
        ]
    ] = []
    for track_index, track in enumerate(tracks):
        iterator = iter(track)
        try:
            message = next(iterator)
        except StopIteration:
            continue
        heappush(
            heap,
            (message.time, track_index, 0, message, iterator),
        )

    while heap:
        absolute_tick, track_index, event_index, message, iterator = heappop(heap)
        yield absolute_tick, message
        try:
            following = next(iterator)
        except StopIteration:
            continue
        heappush(
            heap,
            (
                absolute_tick + following.time,
                track_index,
                event_index + 1,
                following,
                iterator,
            ),
        )


def _preflight_midi(data: bytes, *, max_notes: int) -> None:
    if len(data) < 14 or data[:4] != b"MThd":
        return
    header_length = int.from_bytes(data[4:8], "big")
    if header_length < 6 or 8 + header_length > len(data):
        raise MidiConversionError("The uploaded file is not valid MIDI.")
    track_count = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")
    if division == 0 or division & 0x8000:
        raise MidiConversionError(
            "SMPTE or zero MIDI time division is unsupported; "
            "use a positive ticks-per-beat value."
        )

    offset = 8 + header_length
    note_count = 0
    event_count = 0
    max_events = max(1_000, max_notes * 16)
    for _ in range(track_count):
        if offset + 8 > len(data) or data[offset : offset + 4] != b"MTrk":
            raise MidiConversionError("The uploaded file is not valid MIDI.")
        track_length = int.from_bytes(data[offset + 4 : offset + 8], "big")
        track_start = offset + 8
        track_end = track_start + track_length
        if track_end > len(data):
            raise MidiConversionError("The uploaded file is not valid MIDI.")
        notes, events = _scan_track_events(
            memoryview(data)[track_start:track_end],
            max_notes=max_notes - note_count,
            max_events=max_events - event_count,
        )
        note_count += notes
        event_count += events
        offset = track_end


def _scan_track_events(
    track: memoryview,
    *,
    max_notes: int,
    max_events: int,
) -> tuple[int, int]:
    position = 0
    running_status: int | None = None
    note_count = 0
    event_count = 0
    while position < len(track):
        _, position = _read_variable_length(track, position)
        if position >= len(track):
            raise MidiConversionError("The uploaded file is not valid MIDI.")

        status = track[position]
        if status & 0x80:
            position += 1
            if status < 0xF0:
                running_status = status
            else:
                running_status = None
        elif running_status is not None:
            status = running_status
        else:
            raise MidiConversionError("The uploaded file is not valid MIDI.")

        event_count += 1
        if event_count > max_events:
            raise MidiConversionError("MIDI contains too many events.")
        if status == 0xFF:
            if position >= len(track):
                raise MidiConversionError("The uploaded file is not valid MIDI.")
            position += 1
            length, position = _read_variable_length(track, position)
            position += length
        elif status in {0xF0, 0xF7}:
            length, position = _read_variable_length(track, position)
            position += length
        elif 0x80 <= status <= 0xEF:
            message_type = status >> 4
            data_length = 1 if message_type in {0xC, 0xD} else 2
            if position + data_length > len(track):
                raise MidiConversionError("The uploaded file is not valid MIDI.")
            if message_type == 0x9 and track[position + 1] > 0:
                note_count += 1
                if note_count > max_notes:
                    raise MidiConversionError("MIDI contains too many notes.")
            position += data_length
        else:
            data_length = {
                0xF1: 1,
                0xF2: 2,
                0xF3: 1,
            }.get(status, 0)
            position += data_length
        if position > len(track):
            raise MidiConversionError("The uploaded file is not valid MIDI.")
    return note_count, event_count


def _read_variable_length(data: memoryview, position: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if position >= len(data):
            raise MidiConversionError("The uploaded file is not valid MIDI.")
        byte = data[position]
        position += 1
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value, position
    raise MidiConversionError("The uploaded file is not valid MIDI.")


def _validate_export_parameters(
    *,
    tempo: float,
    time_signature: TimeSignature,
) -> None:
    if not isfinite(tempo) or not 4.0 <= tempo <= 400.0:
        raise MidiConversionError("MIDI tempo must be between 4 and 400 BPM.")
    denominator = time_signature.denominator
    if (
        time_signature.numerator > 255
        or denominator > 128
        or denominator & (denominator - 1)
    ):
        raise MidiConversionError(
            "MIDI time signature requires numerator <= 255 and "
            "a power-of-two denominator <= 128."
        )


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
