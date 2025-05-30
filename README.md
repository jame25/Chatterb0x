# Chatterb0x Clipboard Reader (PySide6 Edition)

<p align="center">
  <img src="app_icon.png" alt="Chatterb0x Clipboard Reader Logo" width="200"/>
</p>

A system tray utility that leverages [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by [Resemble AI](https://www.resemble.ai/) to read clipboard text aloud. It offers on-demand reading, automatic reading of new clipboard content, and saving text as WAV audio files. This version uses PySide6 for the GUI and PyAudio for robust audio playback.

## ✨ Features

*   **Clipboard to Speech**: Reads the current clipboard text aloud.
*   **Auto-Read Mode**: Automatically reads new text copied to the clipboard.
    *   Skips URLs and file paths to avoid unintended reading.
*   **Save to WAV**: Converts clipboard text to speech and saves it as a `.wav` file.
*   **Voice Selection**:
    *   Choose between different voice prompts (e.g., male, female).
    *   Option to use the model's default voice without a specific prompt.
    *   Attempts to auto-generate basic prompts if custom ones are not found.
*   **System Tray Integration**: Runs discreetly in the system tray with a context menu for easy access.
*   **GPU Accelerated**: Utilizes CUDA-enabled GPUs for fast TTS generation.
*   **Cross-Platform (Untested, GUI/Core Logic)**: Core Python logic is cross-platform. Setup scripts currently focus on Windows (`run.bat`) and Linux/macOS (`run.sh`).

## 🚀 Getting Started

### Prerequisites

1.  **Python**: Python 3.8 - 3.12 (Python 3.11 is recommended).
    *   Ensure Python is added to your system's PATH during installation.
    *   Download from [python.org](https://www.python.org/downloads/).
2.  **`uv` (Optional but Recommended for Speed)**: A fast Python package installer and resolver.
    *   Installation: `pip install uv` or see [uv's official installation guide](https://github.com/astral-sh/uv#installation).
    *   The setup scripts will attempt to help you install `uv` if it's missing.
3.  **NVIDIA GPU & CUDA (for TTS)**:
    *   An NVIDIA GPU is **required** for `Chatterbox TTS` to function.
    *   Ensure you have the latest NVIDIA drivers installed.
    *   The `install_torch.py` script (run by the setup scripts) will attempt to install the correct PyTorch version with CUDA support. If CUDA is not detected, it may fall back to a CPU version of PyTorch, but **Chatterbox TTS itself primarily targets GPU usage.**

### Installation & Running

This project includes one-click installer scripts (`run.bat` for Windows, `run.sh` for Linux/macOS) to simplify setup.

<details>
<summary><strong>Windows (using <code>run.bat</code>)</strong></summary>

1.  **Download/Clone:** Get the project files.
2.  **Run Installer:** Double-click `run.bat`.
    *   It will check for Python and `uv`.
    *   It will create a virtual environment (e.g., in `.venv`).
    *   It will install dependencies using `uv` from `requirements.lock.txt` (or compile it from `requirements.in`).
    *   It will then run `install_torch.py` to install the appropriate PyTorch version for your CUDA setup (or CPU if CUDA is not found/supported by the script's detection).
    *   Finally, it will launch the application (`main.py` or your script name).
3.  The application will appear as an icon in your system tray.

</details>

<details>
<summary><strong>Linux/macOS (using <code>run.sh</code> - *you'll need to create this based on run.bat*)</strong></summary>

1.  **Download/Clone:** Get the project files.
2.  **Make Executable (if needed):** `chmod +x run.sh`
3.  **Run Installer:** Execute `./run.sh` from your terminal.
    *   The script should perform similar steps to `run.bat`:
        *   Check for Python and `uv`.
        *   Create a virtual environment.
        *   Install dependencies.
        *   Run `install_torch.py`.
        *   Launch the application.
4.  The application will appear as an icon in your system tray (behavior might vary slightly depending on your desktop environment).

</details>

<details>
<summary><strong>Manual Setup (if not using installer scripts)</strong></summary>

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jame25/Chatterb0x
    cd Chatterb0x
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # Using uv (recommended)
    uv venv .venv --python 3.11 # Or your preferred Python version
    source .venv/bin/activate   # Linux/macOS
    # .venv\Scripts\activate    # Windows

    # Or using Python's built-in venv
    # python -m venv .venv
    # source .venv/bin/activate   # Linux/macOS
    # .venv\Scripts\activate    # Windows
    ```
3.  **Install dependencies:**
    *   If `requirements.lock.txt` exists and is up-to-date:
        ```bash
        uv pip sync requirements.lock.txt
        ```
    *   Otherwise, or to regenerate the lock file from `requirements.in`:
        ```bash
        uv pip compile requirements.in -o requirements.lock.txt
        uv pip sync requirements.lock.txt
        ```
    *   If not using `uv`:
        ```bash
        # This might be slower and less deterministic than using a lock file with uv
        pip install -r requirements.in 
        ```
4.  **Install PyTorch:**
    Execute the `install_torch.py` script to get the correct PyTorch version for your system (CUDA or CPU).
    ```bash
    python install_torch.py
    ```
5.  **Run the application:**
    ```bash
    python main.py # Or your main script name
    ```

</details>

### Configuration

*   **Voice Prompts**:
    *   Place your custom voice prompt audio files (e.g., short WAV files, ideally around 3-10 seconds) named `male_voice.wav` and `female_voice.wav` into a `voice_prompts` directory within the application's root folder.
    *   If these files are not found, the application will attempt to generate basic default prompts using the TTS model itself. The quality of these auto-generated prompts will depend on the model's default voice.
*   **Icon**: Ensure an icon file (e.g., `app_icon.png`) is present in the application's root directory for the system tray icon.

## 🛠️ Usage

*   **Right-click** the system tray icon to access the menu.
*   **Read Clipboard**: Reads the current text content of the clipboard.
*   **Save to WAV**: Saves the current clipboard text as a WAV audio file. You'll be prompted to choose a save location.
*   **Stop Reading**: Immediately stops any ongoing text-to-speech playback.
*   **Voice**: Select your preferred voice prompt (Male, Female, or Model Default).
*   **Auto-Read Clipboard**: Toggle this option to enable/disable automatic reading of newly copied text.
*   **Quit**: Exits the application.

## 📄 License

This project is based on initial work found at [https://github.com/jame25/Chatterb0x](https://github.com/jame25/Chatterb0x).
This modified version is distributed under the **MIT License**. See the `LICENSE` file for more details.

The core TTS functionality relies on [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by [Resemble AI](https://www.resemble.ai/).
This project also utilizes:
*   [PySide6 (LGPL v3)](https://www.qt.io/licensing/) for the graphical user interface.
*   [PyAudio (MIT License)](https://people.csail.mit.edu/hubert/pyaudio/) for audio playback.
*   [uv (MIT/Apache 2.0 License)](https://github.com/astral-sh/uv) for Python packaging (in setup scripts).
*   [PyTorch (BSD-style License)](https://github.com/pytorch/pytorch/blob/master/LICENSE).

## 🤝 Contributing

Contributions are welcome! If you have improvements, bug fixes, or new features, please feel free to:
1.  Fork the repository.
2.  Create a new branch for your changes.
3.  Submit a Pull Request with a clear description of your work.

## 🐛 Issues
If you encounter any issues, please open an issue on the GitHub repository. Include as much detail as possible to help us diagnose and fix the problem.
### Known issues:
*   **Clipboard Reading**: The application may not read clipboard content immediately if the text is copied too quickly after the last read. This is a limitation of the clipboard monitoring mechanism.
*   **Voice Prompt Generation**: Auto-generated voice prompts may not sound as good as custom ones. It's recommended to provide your own short audio files for better quality.
*   **System Tray Icon**: The system tray icon may not appear on some Linux desktop environments. This is a known limitation and may require additional configuration or a different library for better compatibility.
* **The initial generated audio will be male voices**: The first time you run the application, the generated audio will default to male voices (due to chatterbox's default behavior). Subsequent runs **OR** replacing while running will respect the new one if you have provided custom `wav` files in the `voice_prompts` named `female_voice.wav` and `male_voice.wav` directory.

## 🙏 Support the Original Author

Original Author's Support:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/jame25)

If you find this PySide6 version and its improvements useful, consider starring the repository!

---