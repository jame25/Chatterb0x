# ChatterbOx


## Description

This is a tray utility that supports [Chatterbox TTS](https://github.com/resemble-ai/chatterbox). It can automatically read aloud new text copied to the clipboard or read the current clipboard content on demand. It also allows saving the clipboard text as a WAV audio file.

## Features

*   Read clipboard text aloud using Text-to-Speech.
*   Automatic reading of new clipboard content.
*   Save clipboard text as a WAV audio file.
*   Select between different voice prompts (male/female).
*   System tray integration for easy access to features.

## Requirements

*   Python 3.x
*   Required Python packages: `PyQt5`, `pyperclip`, `torch`, `torchaudio`, `pyaudio`, `chatterbox-tts`, `numpy`, `pathlib`, `queue`, `re`, `os`, `time`, `logging`.
*   Voice prompt audio files (`male_voice.wav`, `female_voice.wav`) in a `voice_prompts` directory (the application will attempt to create these if they don't exist, but can include your own short (30s) voice samples).
*   The system tray icon file named `app_icon.png` in the application directory.
