class FakeAdapter:
    def is_available(self) -> bool:
        return True

    def render(
        self,
        *,
        voices: object,
        tempo: float,
        timbre: str | None,
        guide_audio: bytes,
        sample_rate: int,
    ) -> bytes:
        del voices, tempo, timbre, sample_rate
        return guide_audio


adapter = FakeAdapter()
