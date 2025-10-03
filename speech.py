from typing import Optional
import io
import os

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

AUDIO_FORMAT = "wav"


def _openai_client() -> Optional["OpenAI"]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)  # type: ignore
    except Exception:
        return None


def transcribe_audio_wav(audio_bytes: bytes, source_lang: str = "en") -> Optional[str]:
    """Transcribe WAV audio bytes to text using OpenAI Whisper if available.

    Returns None if unavailable or on failure.
    """
    client = _openai_client()
    if client is None:
        return None

    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"input.{AUDIO_FORMAT}"
        try:
            resp = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file,
                language=source_lang,
            )
            return (getattr(resp, "text", "") or "").strip()
        except Exception:
            audio_file.seek(0)
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=source_lang,
            )
            return (getattr(resp, "text", "") or "").strip()
    except Exception:
        return None