from typing import Optional
import os

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from deep_translator import GoogleTranslator
except ImportError:  # pragma: no cover
    GoogleTranslator = None  # type: ignore

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
    "ja": "Japanese",
}

LANG_NAME_TO_CODE = {v: k for k, v in SUPPORTED_LANGUAGES.items()}

# deep_translator language code mapping adjustments
DEEP_TRANSLATOR_CODE_MAP = {
    "zh": "zh-CN",
}

def _openai_client() -> Optional["OpenAI"]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)  # type: ignore
    except Exception:
        return None

def _normalize_lang(lang: Optional[str]) -> str:
    lang = (lang or "").lower()
    if lang in SUPPORTED_LANGUAGES:
        return lang
    return "en"

def translate_text(text: str, src_lang: Optional[str] = None, target_lang: Optional[str] = None) -> str:
    if not text:
        return ""

    src = _normalize_lang(src_lang or "en")
    dst = _normalize_lang(target_lang or "hi")

    # Try OpenAI translation first for quality
    client = _openai_client()
    if client is not None:
        try:
            system_prompt = (
                f"You are a professional translation engine. Translate the user text from "
                f"{SUPPORTED_LANGUAGES[src]} ({src}) to {SUPPORTED_LANGUAGES[dst]} ({dst}). "
                "Only return the translated text without any additional commentary."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
            )
            output = resp.choices[0].message.content or ""
            return output.strip()
        except Exception:
            pass

    # Fallback to deep-translator's GoogleTranslator
    if GoogleTranslator is not None:
        try:
            src_dt = DEEP_TRANSLATOR_CODE_MAP.get(src, src)
            dst_dt = DEEP_TRANSLATOR_CODE_MAP.get(dst, dst)
            translator = GoogleTranslator(source=src_dt, target=dst_dt)
            result = translator.translate(text)
            return result.strip() if result else ""
        except Exception:
            pass

    # Last fallback: return original text
    return text
