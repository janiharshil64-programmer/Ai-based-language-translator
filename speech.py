from typing import Optional
import io
import os

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
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
    """
    Transcribe WAV audio bytes to text using OpenAI Whisper.

    Returns transcribed text or None if unavailable or on error.
    """
    client = _openai_client()
    if client is None:
        return None

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = f"input.{AUDIO_FORMAT}"

    try:
        response = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language=source_lang,
        )
        return response.text.strip()
    except Exception:
        try:
            audio_file.seek(0)
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=source_lang,
            )
            return response.text.strip()
        except Exception:
            return None
