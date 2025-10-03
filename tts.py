from typing import Optional
import io
import os

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from gtts import gTTS
except ImportError:  # pragma: no cover
    gTTS = None  # type: ignore

LANG_TO_TTS_VOICE = {
    "en": "alloy",
    "hi": "alloy",
    "fr": "alloy",
    "de": "alloy",
    "es": "alloy",
    "zh": "alloy",
    "ja": "alloy",
}

def _openai_client() -> Optional["OpenAI"]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)  # type: ignore
    except Exception:
        return None

def synthesize_speech(text: str, language: str = "en") -> Optional[bytes]:
    if not text:
        return None

    lang = (language or "en").lower()

    client = _openai_client()
    if client is not None:
        try:
            voice = LANG_TO_TTS_VOICE.get(lang, "alloy")
            # Try non-streaming OpenAI TTS
            try:
                resp = client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice=voice,
                    format="mp3",
                    input=text,
                )
                if hasattr(resp, "read"):
                    return resp.read()
                if hasattr(resp, "content") and isinstance(resp.content, (bytes, bytearray)):
                    return bytes(resp.content)
            except Exception:
                pass

            # Try streaming variant if available
            try:
                stream_ctx = getattr(client.audio.speech, "with_streaming_response", None)
                if stream_ctx is not None:
                    with stream_ctx.create(model="gpt-4o-mini-tts", voice=voice, format="mp3", input=text) as stream:
                        buf = io.BytesIO()
                        if hasattr(stream, "get_chunks"):
                            for chunk in stream.get_chunks():
                                buf.write(chunk)
                            return buf.getvalue()
            except Exception:
                pass
        except Exception:
            pass

    # Fallback to gTTS for TTS (requires internet)
    if gTTS is not None:
        try:
            mp3_bytes = io.BytesIO()
            gtts_lang = "zh-cn" if lang == "zh" else lang  # gTTS expects "zh-cn"
            gTTS(text=text, lang=gtts_lang, slow=False).write_to_fp(mp3_bytes)
            return mp3_bytes.getvalue()
        except Exception:
            return None

    return None
