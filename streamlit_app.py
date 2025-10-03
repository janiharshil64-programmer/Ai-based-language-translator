import time
from typing import Tuple

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from translator import translate_text, SUPPORTED_LANGUAGES, LANG_NAME_TO_CODE
from speech import transcribe_audio_wav
from tts import synthesize_speech

APP_TITLE = "Realtime Multi‑Language Translator"

def init_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🌐", layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Translate between English, Hindi, French, German, Spanish, Chinese, and Japanese."
    )
    if "history" not in st.session_state:
        st.session_state.history = []

def lang_selectors() -> Tuple[str, str]:
    names = list(SUPPORTED_LANGUAGES.values())
    code_to_name = SUPPORTED_LANGUAGES

    left, right = st.columns(2)
    with left:
        default_src_name = code_to_name["en"]
        src_name = st.selectbox(
            "Source language",
            names,
            index=names.index(default_src_name),
            key="src_lang_select",
        )
    with right:
        default_dst_name = code_to_name["hi"]
        dst_name = st.selectbox(
            "Target language",
            names,
            index=names.index(default_dst_name),
            key="dst_lang_select",
        )
    return LANG_NAME_TO_CODE[src_name], LANG_NAME_TO_CODE[dst_name]

def text_translation_tab() -> None:
    src_code, dst_code = lang_selectors()
    st.subheader("Text translation")
    input_text = st.text_area("Enter text", height=150, key="text_input")
    col1, col2 = st.columns([1, 3])
    with col1:
        translate_clicked = st.button("Translate", type="primary")
    with col2:
        stream_simulated = st.toggle("Simulate streaming output", value=True)

    if translate_clicked and input_text.strip():
        with st.spinner("Translating..."):
            result = translate_text(input_text.strip(), src_lang=src_code, target_lang=dst_code)
            output_container = st.empty()
            if stream_simulated:
                acc = ""
                for token in result.split():
                    acc += (" " if acc else "") + token
                    output_container.write(acc)
                    time.sleep(0.02)
            else:
                output_container.write(result)

            st.session_state.history.append((src_code, dst_code, input_text, result))

        st.divider()
        st.markdown("**Play translation**")
        voice_enabled = st.toggle("Text-to-speech", value=False, key="tts_toggle_text")
        if voice_enabled:
            voice_audio = synthesize_speech(result, language=dst_code)
            if voice_audio:
                st.audio(voice_audio, format="audio/mp3")
            else:
                st.info("TTS not available. Configure OPENAI_API_KEY for best quality.")

def speech_translation_tab() -> None:
    src_code, dst_code = lang_selectors()
    st.subheader("Speech translation (record and translate)")
    st.caption("Record speech, we transcribe and translate it. For best results, set OPENAI_API_KEY.")

    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop",
        key="recorder",
        just_once=False,
    )

    if audio and audio.get("bytes"):
        st.audio(audio["bytes"], format="audio/wav")
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio_wav(audio["bytes"], source_lang=src_code)
            if transcript:
                st.markdown("**Transcript**: " + transcript)
                with st.spinner("Translating..."):
                    translated = translate_text(transcript, src_lang=src_code, target_lang=dst_code)
                    st.markdown("**Translation**: " + translated)

                    play_tts = st.toggle("Play translated audio", value=False, key="tts_toggle_speech")
                    if play_tts:
                        voice_audio = synthesize_speech(translated, language=dst_code)
                        if voice_audio:
                            st.audio(voice_audio, format="audio/mp3")
                        else:
                            st.info("TTS not available. Configure OPENAI_API_KEY for best quality.")
            else:
                st.warning("Transcription unavailable. Provide OPENAI_API_KEY for speech.")

def history_panel() -> None:
    st.subheader("History")
    if not st.session_state.history:
        st.write("No translations yet.")
        return
    for src, dst, src_text, dst_text in reversed(st.session_state.history[-20:]):
        st.write(f"{SUPPORTED_LANGUAGES[src]} → {SUPPORTED_LANGUAGES[dst]}")
        st.code(src_text)
        st.write(dst_text)
        st.divider()

def main() -> None:
    init_page()
    tabs = st.tabs(["Text", "Speech", "History"])
    with tabs[0]:
        text_translation_tab()
    with tabs[1]:
        speech_translation_tab()
    with tabs[2]:
        history_panel()

if __name__ == "__main__":
    main()


