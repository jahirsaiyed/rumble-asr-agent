"""Split a token stream into speakable sentences so TTS can start early."""
import re

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_FIRST = 25  # flush the first sentence quickly, but not on "Hi."


class SentenceChunker:
    def __init__(self) -> None:
        self._buf = ""
        self._emitted = 0

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            m = _BOUNDARY.search(self._buf)
            if not m:
                break
            sentence = self._buf[: m.start() + (1 if self._buf[m.start()] in ".!?" else 0)].strip()
            minimum = _MIN_FIRST if self._emitted == 0 else 1
            if len(sentence) < minimum and m.end() < len(self._buf):
                # too short to speak alone, look for the next boundary
                nxt = _BOUNDARY.search(self._buf, m.end())
                if not nxt:
                    break
            self._buf = self._buf[m.end():]
            if sentence:
                out.append(sentence)
                self._emitted += 1
        return out

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []
