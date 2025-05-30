import sys
import queue
import torchaudio as ta
import torch
import numpy as np
from pathlib import Path
import re
import logging
import urllib.parse
import pyaudio  # ADDED FOR PYAUDIO

from PySide6.QtWidgets import (
    QApplication, QSystemTrayIcon, QMenu,
    QMessageBox, QFileDialog
)
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Signal, QObject, QRunnable, QThreadPool, QThread

# Graceful ChatterboxTTS import
try:
    from chatterbox.tts import ChatterboxTTS
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    ChatterboxTTS = None
    print("WARNING: chatterbox.tts module not found. TTS functionality will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clipboard_reader_pyside6_pyaudio.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Utility Functions ---


def split_text_into_chunks(text, max_chars_per_chunk=200, sentence_delimiters=r'(?<=[.!?…])\s+'):
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    sentences = re.split(sentence_delimiters, text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not current_chunk:
            current_chunk = sentence
        elif len(current_chunk) + 1 + len(sentence) <= max_chars_per_chunk:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars_per_chunk * 1.5:  # Allow some overflow
            words, temp_chunk = chunk.split(' '), ""
            for word in words:
                if len(temp_chunk) + 1 + len(word) <= max_chars_per_chunk:
                    if temp_chunk:
                        temp_chunk += " " + word
                    else:
                        temp_chunk = word
                else:
                    if temp_chunk:
                        final_chunks.append(temp_chunk)
                    temp_chunk = word
            if temp_chunk:
                final_chunks.append(temp_chunk)
        else:
            final_chunks.append(chunk)
    return final_chunks if final_chunks else [text]


# --- QThread for Audio Generation ---
class AudioGenerator(QThread):
    audio_chunk_ready = Signal(np.ndarray)  # Emits float32 numpy array
    generation_finished = Signal()
    error_occurred = Signal(str)
    progress_update = Signal(str)

    def __init__(self, model, text_queue: queue.Queue, voice_prompt_path: str = None, parent=None):
        super().__init__(parent)
        self.model = model
        self.text_queue = text_queue
        self.voice_prompt_path = voice_prompt_path
        self._is_running = True

    def run(self):
        logger.debug("AudioGenerator: Thread started.")
        if not CHATTERBOX_AVAILABLE or not self.model:
            logger.error("AudioGenerator: ChatterboxTTS model not available.")
            self.error_occurred.emit("TTS Model not available.")
            self.generation_finished.emit()
            return
        try:
            while self._is_running:
                try:
                    text_chunk = self.text_queue.get(timeout=0.1)
                    if text_chunk is None:
                        logger.debug(
                            "AudioGenerator: Received None, stopping.")
                        break
                    if not self._is_running:
                        logger.debug("AudioGenerator: Stop signal received.")
                        break

                    logger.info(
                        f"AudioGenerator: Generating for: '{text_chunk[:50]}...'")
                    self.progress_update.emit(
                        f"Generating: {text_chunk[:30]}...")

                    if self.voice_prompt_path and Path(self.voice_prompt_path).exists():
                        wav = self.model.generate(
                            text_chunk, audio_prompt_path=self.voice_prompt_path, temperature=0.7, cfg_weight=0.3)
                    else:
                        if self.voice_prompt_path:
                            logger.warning(
                                f"Voice prompt {self.voice_prompt_path} not found.")
                        wav = self.model.generate(
                            text_chunk, temperature=0.7, cfg_weight=0.3)

                    if isinstance(wav, torch.Tensor):
                        audio_data_np = wav.cpu().numpy()
                    else:
                        audio_data_np = np.array(wav)

                    audio_data_np = audio_data_np.astype(
                        np.float32)  # Ensure float32

                    # Ensure mono
                    if audio_data_np.ndim > 1:
                        # Assuming shape [channels, samples] or [samples, channels]
                        # Take the first channel or average if stereo, common for TTS outputs.
                        # If your model always outputs mono, this might not be strictly needed.
                        # [channels, samples] and stereo+
                        if audio_data_np.shape[0] < audio_data_np.shape[1] and audio_data_np.shape[0] > 1:
                            # Take first channel
                            audio_data_np = audio_data_np[0, :]
                        # [samples, channels] and stereo+
                        elif audio_data_np.shape[1] < audio_data_np.shape[0] and audio_data_np.shape[1] > 1:
                            # Take first channel
                            audio_data_np = audio_data_np[:, 0]
                        # If already mono but with an extra dimension [1, samples] or [samples, 1], squeeze it.
                        audio_data_np = audio_data_np.squeeze()

                    self.audio_chunk_ready.emit(audio_data_np)
                    logger.debug(
                        f"AudioGenerator: Emitted audio chunk of shape {audio_data_np.shape}")
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"AudioGenerator: Error: {e}", exc_info=True)
                    self.error_occurred.emit(f"Generation error: {str(e)}")
                    self._is_running = False
                    break
        except Exception as e:
            logger.error(f"AudioGenerator: Fatal error: {e}", exc_info=True)
            self.error_occurred.emit(f"Fatal generator error: {str(e)}")
        finally:
            self.progress_update.emit("TTS Idle")
            self.generation_finished.emit()
            logger.debug("AudioGenerator: Thread finished.")

    def stop(self):
        logger.debug("AudioGenerator: Stop requested.")
        self._is_running = False
        try:
            self.text_queue.put_nowait(None)
        except queue.Full:
            logger.warning("AudioGenerator: Text queue full on stop signal.")
        except Exception as e:
            logger.error(
                f"AudioGenerator: Error sending stop to text queue: {e}")

# --- QThread for PyAudio Playback ---


class AudioPlayerPyAudio(QThread):
    playback_finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, p_audio_instance: pyaudio.PyAudio, audio_data_queue: queue.Queue, sample_rate: int, parent=None):
        super().__init__(parent)
        self.p_audio = p_audio_instance
        self.audio_data_queue = audio_data_queue
        self.sample_rate = sample_rate
        self._is_running = True
        self.stream = None
        self.PYAUDIO_CHUNK_SIZE = 1024

    def run(self):
        logger.debug("AudioPlayerPyAudio: Thread started.")
        try:
            self.stream = self.p_audio.open(format=pyaudio.paFloat32,
                                            channels=1,
                                            rate=self.sample_rate,
                                            output=True,
                                            frames_per_buffer=self.PYAUDIO_CHUNK_SIZE)
            logger.info(
                f"AudioPlayerPyAudio: PyAudio stream opened at {self.sample_rate} Hz, Float32.")
            self.stream.start_stream()

            while self._is_running:
                try:
                    audio_chunk_np = self.audio_data_queue.get(timeout=0.1)
                    if audio_chunk_np is None:
                        logger.debug(
                            "AudioPlayerPyAudio: Received None, stopping playback loop.")
                        break
                    if not self._is_running:
                        logger.debug(
                            "AudioPlayerPyAudio: Stop signal received during playback loop.")
                        break

                    # audio_chunk_np is expected to be float32 numpy array from AudioGenerator
                    self.stream.write(audio_chunk_np.tobytes())

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(
                        f"AudioPlayerPyAudio: Playback error: {e}", exc_info=True)
                    self.error_occurred.emit(f"Playback error: {str(e)}")
                    self._is_running = False
                    break

            # After loop (None received or stop signal), wait for buffer to clear if stream is still active
            if self.stream and self.stream.is_active():
                logger.debug(
                    "AudioPlayerPyAudio: Playback loop ended, waiting for stream buffer to clear.")
                # A more robust wait loop might be needed if latency is an issue
                # For now, relying on stream.stop_stream() to handle flushing/stopping
                # while self.stream.is_active() and self._is_running: # Don't wait if stop() was called
                #     time.sleep(0.01) # Short sleep to yield
                # logger.debug("AudioPlayerPyAudio: Stream buffer likely cleared or stopped.")
                pass  # stop_stream will handle this

        except Exception as e:  # Catch errors from p_audio.open or stream management
            logger.error(
                f"AudioPlayerPyAudio: Stream management error: {e}", exc_info=True)
            self.error_occurred.emit(f"Fatal player error: {str(e)}")
        finally:
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                    logger.debug("AudioPlayerPyAudio: PyAudio stream closed.")
                except Exception as e_close:
                    logger.error(
                        f"AudioPlayerPyAudio: Error closing stream: {e_close}", exc_info=True)

            self.playback_finished.emit()  # Emit finished signal regardless of how it ended
            logger.debug("AudioPlayerPyAudio: Thread finished.")

    def stop(self):
        logger.debug("AudioPlayerPyAudio: Stop requested.")
        self._is_running = False
        try:
            self.audio_data_queue.put_nowait(None)  # Ensure get() unblocks
        except queue.Full:
            logger.warning(
                "AudioPlayerPyAudio: Audio data queue full on stop signal.")
        except Exception as e:
            logger.error(
                f"AudioPlayerPyAudio: Error sending stop to audio data queue: {e}")


# --- QRunnable for Saving to WAV ---
class SaveToWavTask(QRunnable):
    class Signals(QObject):
        finished = Signal(str)
        error = Signal(str)

    def __init__(self, model, text, file_path, voice_prompt_path=None):
        super().__init__()
        self.model, self.text, self.file_path, self.voice_prompt_path = model, text, file_path, voice_prompt_path
        self.signals = SaveToWavTask.Signals()
        self.setAutoDelete(True)

    def run(self):
        logger.info(f"SaveToWavTask: Starting for '{self.file_path}'")
        if not CHATTERBOX_AVAILABLE or not self.model:
            logger.error("SaveToWavTask: TTS model not available.")
            self.signals.error.emit("TTS Model not available for saving.")
            return
        try:
            if self.voice_prompt_path and Path(self.voice_prompt_path).exists():
                wav = self.model.generate(
                    self.text, audio_prompt_path=self.voice_prompt_path)
            else:
                if self.voice_prompt_path:
                    logger.warning(
                        f"Voice prompt {self.voice_prompt_path} not found for saving.")
                wav = self.model.generate(self.text)

            if not isinstance(wav, torch.Tensor):
                wav = torch.tensor(wav, dtype=torch.float32)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)  # Add channel dim for torchaudio

            # Ensure tensor is on CPU
            ta.save(self.file_path, wav.cpu(), self.model.sr)
            logger.info(
                f"SaveToWavTask: Successfully saved to '{self.file_path}'")
            self.signals.finished.emit(self.file_path)
        except Exception as e:
            logger.error(f"SaveToWavTask: Error: {e}", exc_info=True)
            self.signals.error.emit(f"Failed to save WAV: {str(e)}")

# --- Main Application Logic ---


class ClipboardReader(QObject):
    def __init__(self, app_instance):
        super().__init__()
        self.app = app_instance
        self.model = None
        self.audio_generator_thread = None
        self.audio_player_thread = None
        self.p_audio_instance = None  # For PyAudio
        self._is_reading_active = False
        self._generation_complete_flag = False

        self.text_processing_queue = queue.Queue(
            maxsize=100)  # For text chunks to generator
        # For audio chunks (np.array) to player
        self.audio_playback_queue = queue.Queue(maxsize=50)

        self.last_clipboard_text = ""
        try:
            self.last_clipboard_text = self.app.clipboard().text()
        except Exception as e:
            logger.warning(f"Could not get initial clipboard text: {e}")

        self.url_pattern = re.compile(
            r'^(file:///|https?://).+', re.IGNORECASE)
        self.file_path_pattern = re.compile(
            r'^[a-zA-Z]:\\|^/|^\\\\|^file:///', re.IGNORECASE)
        self.thread_pool = QThreadPool.globalInstance()

        self._setup_ui()

        if CHATTERBOX_AVAILABLE:
            try:
                self.p_audio_instance = pyaudio.PyAudio()
                logger.info("PyAudio instance created successfully.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize PyAudio: {e}", exc_info=True)
                QMessageBox.critical(
                    None, "Audio Error", "Failed to initialize PyAudio. Playback will be disabled.")
                self.p_audio_instance = None  # Ensure it's None if init failed

            if self.p_audio_instance:  # Only load model if PyAudio is also ready
                self.initialize_model_async()
            else:  # PyAudio failed, disable TTS related UI
                self.tray_icon.setToolTip("Clipboard TTS Reader - Audio Error")
                self.read_action.setEnabled(False)
                self.save_action.setEnabled(False)
                self.voice_menu.setEnabled(False)
                self.auto_read_action.setEnabled(False)
        else:
            logger.warning(
                "ChatterboxTTS library not available. TTS features disabled.")
            self.tray_icon.showMessage(
                "TTS Error", "TTS library not found.", QSystemTrayIcon.Critical, 5000)

        self.app.clipboard().dataChanged.connect(self.on_clipboard_changed)

    def _setup_ui(self):
        logger.debug("Setting up UI...")
        icon_path_str = 'app_icon.png'
        icon = QIcon.fromTheme("emblem-sound", QIcon(icon_path_str))
        if QIcon(icon_path_str).isNull() and icon.isNull():
            logger.warning(
                f"Icon file {icon_path_str} not found and no theme icon 'emblem-sound'.")

        self.tray_icon = QSystemTrayIcon(icon, self)
        self.tray_icon.setToolTip("Clipboard TTS Reader - Initializing...")
        self.tray_menu = QMenu()

        self.read_action = QAction('Read Clipboard', self)
        self.read_action.triggered.connect(self.trigger_read_clipboard)
        self.read_action.setEnabled(False)
        self.tray_menu.addAction(self.read_action)
        self.save_action = QAction('Save to WAV', self)
        self.save_action.triggered.connect(self.save_to_wav)
        self.save_action.setEnabled(False)
        self.tray_menu.addAction(self.save_action)
        self.stop_action = QAction('Stop Reading', self)
        self.stop_action.triggered.connect(self.stop_reading)
        self.stop_action.setEnabled(False)
        self.tray_menu.addAction(self.stop_action)
        self.tray_menu.addSeparator()

        self.voice_menu = QMenu('Voice', self.tray_menu)
        self.current_voice_type = 'female'
        self.voice_prompts_dir = Path('voice_prompts')
        self.voice_prompts_dir.mkdir(parents=True, exist_ok=True)
        self.voice_prompt_paths = {'female': self.voice_prompts_dir/'female_voice.wav',
                                   'male': self.voice_prompts_dir/'male_voice.wav', 'default': None}
        self.voice_action_group = []

        def add_voice_action(name, key):
            act = QAction(name, self.voice_menu, checkable=True)
            act.triggered.connect(
                lambda c, k=key: self.set_voice_type(k) if c else None)
            if key == self.current_voice_type:
                act.setChecked(True)
            self.voice_menu.addAction(act)
            self.voice_action_group.append(act)
        add_voice_action('Female (Prompt)', 'female')
        add_voice_action('Male (Prompt)', 'male')
        add_voice_action('Model Default', 'default')
        self.voice_menu.setEnabled(False)
        self.tray_menu.addMenu(self.voice_menu)
        self.tray_menu.addSeparator()

        self.auto_read_action = QAction(
            'Auto-Read Clipboard', self, checkable=True)
        self.auto_read_action.setChecked(False)
        self.auto_read_action.setEnabled(False)
        self.tray_menu.addAction(self.auto_read_action)
        self.tray_menu.addSeparator()

        quit_action = QAction('Quit', self)
        quit_action.triggered.connect(self.quit_application)
        self.tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.show()
        logger.debug("UI setup complete.")

    def initialize_model_async(self):
        class ModelLoaderTask(QRunnable):
            class Signals(QObject):
                model_ready = Signal(object)
                model_error = Signal(str)

            def __init__(self): super().__init__(
            ); self.signals = ModelLoaderTask.Signals()

            def run(self):
                logger.info("ModelLoaderTask: Starting TTS model init...")
                if not torch.cuda.is_available():
                    self.signals.model_error.emit(
                        "CUDA GPU not available for TTS.")
                    return
                try:
                    model = ChatterboxTTS.from_pretrained(device='cuda')
                    self.signals.model_ready.emit(model)
                except Exception as e:
                    logger.critical(
                        f"ModelLoaderTask: Failed to init TTS model: {e}", exc_info=True)
                    self.signals.model_error.emit(
                        f'Failed to initialize TTS model: {str(e)}')
        self.model_loader_task = ModelLoaderTask()
        self.model_loader_task.signals.model_ready.connect(
            self._on_model_ready)
        self.model_loader_task.signals.model_error.connect(
            self._on_model_load_error)
        self.thread_pool.start(self.model_loader_task)

    def _on_model_ready(self, model_instance):
        self.model = model_instance
        logger.info("TTS Model is ready and assigned.")
        self.tray_icon.setToolTip("Clipboard TTS Reader - Ready")
        self.tray_icon.showMessage(
            "TTS Ready", "Text-to-Speech model loaded.", QSystemTrayIcon.Information, 2000)
        self.read_action.setEnabled(True)
        self.save_action.setEnabled(True)
        self.voice_menu.setEnabled(True)
        self.auto_read_action.setEnabled(True)
        self._create_default_voice_prompts_if_needed()

    def _on_model_load_error(self, error_message):
        logger.error(f"Failed to load TTS model: {error_message}")
        QMessageBox.critical(None, 'Model Error', error_message +
                             "\nTTS functionality will be disabled.")
        self.tray_icon.setToolTip("Clipboard TTS Reader - Model Error")
        # Actions remain disabled

    def _create_default_voice_prompts_if_needed(self):
        if not self.model:
            logger.warning("Model not init, cannot create voice prompts.")
            return
        sample_text = "This is a sample of my voice."
        for key, path in self.voice_prompt_paths.items():
            if key == 'default' or not path:
                continue
            if not path.exists():
                try:
                    logger.info(f"Creating {key} voice prompt at {path}...")
                    wav = self.model.generate(sample_text)
                    if not isinstance(wav, torch.Tensor):
                        wav = torch.tensor(wav, dtype=torch.float32)
                    if wav.ndim == 1:
                        wav = wav.unsqueeze(0)
                    ta.save(str(path), wav.cpu(), self.model.sr)
                    logger.info(f"Created {key} voice prompt.")
                except Exception as e:
                    logger.error(
                        f"Failed to create {key} voice prompt: {e}", exc_info=True)

    def set_voice_type(self, voice_key):
        if self.current_voice_type != voice_key:
            self.current_voice_type = voice_key
            logger.info(f"Voice set to: {voice_key}")
            # Update menu check states (ensure correct logic for matching action text to key)
            for action in self.voice_action_group:
                action_text_prefix = action.text().split(" ")[0].lower()
                is_default_action = "Model Default" in action.text()

                if voice_key == "default" and is_default_action:
                    action.setChecked(True)
                elif action_text_prefix == voice_key and not is_default_action:
                    action.setChecked(True)
                else:
                    action.setChecked(False)
            self.tray_icon.showMessage(
                "Voice Changed", f"Switched to {voice_key} voice.", QSystemTrayIcon.Information, 1500)

    def get_current_voice_prompt_path(self):
        path = self.voice_prompt_paths.get(self.current_voice_type)
        if path and path.exists():
            return str(path)
        elif path:
            logger.warning(
                f"Voice prompt for {self.current_voice_type} ({path}) not found. Using model default.")
        return None

    def is_valid_text_for_reading(self, text):
        s = text.strip()
        if not s:
            return False
        if self.url_pattern.match(s) or self.file_path_pattern.match(s):
            logger.debug(f"Skipping URL/path: {s[:60]}...")
            return False
        try:
            if urllib.parse.urlparse(s).scheme in ('http', 'https', 'file'):
                logger.debug(f"Skipping parsed URL: {s[:60]}...")
                return False
        except ValueError:
            pass
        return True

    def on_clipboard_changed(self):
        if not self.model or not self.p_audio_instance:
            return  # Don't process if model or PyAudio not ready
        try:
            current_text = self.app.clipboard().text()
            if current_text and current_text != self.last_clipboard_text:
                logger.debug(f"Clipboard changed: '{current_text[:60]}...'")
                self.last_clipboard_text = current_text
                if self.auto_read_action.isChecked() and self.is_valid_text_for_reading(current_text):
                    logger.info("Auto-read triggered by clipboard change.")
                    self.process_text_for_reading(current_text)
        except Exception as e:
            logger.error(f"Error in on_clipboard_changed: {e}", exc_info=True)

    def trigger_read_clipboard(self):
        if not self.model:
            self.tray_icon.showMessage(
                "TTS Not Ready", "The TTS model is not loaded.", QSystemTrayIcon.Warning, 2500)
            return
        if not self.p_audio_instance:
            self.tray_icon.showMessage(
                "Audio Error", "PyAudio not initialized.", QSystemTrayIcon.Warning, 2500)
            return

        text = self.app.clipboard().text()
        if not text.strip():
            self.tray_icon.showMessage(
                "Clipboard Empty", "Nothing to read.", QSystemTrayIcon.Warning, 2000)
            return
        if not self.is_valid_text_for_reading(text):
            self.tray_icon.showMessage(
                "Invalid Text", "Content is URL/path.", QSystemTrayIcon.Information, 2500)
            return
        self.process_text_for_reading(text)

    def process_text_for_reading(self, text_to_read: str):
        if not self.model:
            QMessageBox.critical(None, "Error", "TTS Model not initialized.")
            return
        if not self.p_audio_instance:
            QMessageBox.critical(
                None, "Error", "PyAudio not initialized. Cannot play.")
            return

        if self._is_reading_active:
            logger.info("Reading active. Stopping current then starting new.")
            self.stop_reading(silent=True)

        self._is_reading_active = True
        self._generation_complete_flag = False
        self.stop_action.setEnabled(True)

        chunks = split_text_into_chunks(text_to_read)
        if not chunks:
            logger.warning("No processable chunks found in text.")
            self._is_reading_active = False
            self.stop_action.setEnabled(False)
            return

        logger.info(f"Starting to read {len(chunks)} chunks using PyAudio.")

        # Clear queues for new session
        for q in [self.text_processing_queue, self.audio_playback_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Start AudioGenerator
        current_prompt = self.get_current_voice_prompt_path()
        self.audio_generator_thread = AudioGenerator(
            self.model, self.text_processing_queue, current_prompt)
        self.audio_generator_thread.audio_chunk_ready.connect(
            self._on_audio_chunk_generated_for_pyaudio)
        self.audio_generator_thread.generation_finished.connect(
            self._on_generation_thread_finished)
        self.audio_generator_thread.error_occurred.connect(self._on_tts_error)
        self.audio_generator_thread.progress_update.connect(
            self.tray_icon.setToolTip)
        self.audio_generator_thread.start()

        # Start AudioPlayerPyAudio
        self.audio_player_thread = AudioPlayerPyAudio(
            self.p_audio_instance, self.audio_playback_queue, self.model.sr)
        self.audio_player_thread.playback_finished.connect(
            self._on_playback_thread_finished)
        self.audio_player_thread.error_occurred.connect(
            self._on_playback_error)
        self.audio_player_thread.start()

        for chunk in chunks:
            self.text_processing_queue.put(chunk)
        self.text_processing_queue.put(None)  # Signal end to generator

    def _on_audio_chunk_generated_for_pyaudio(self, audio_data_np_float32: np.ndarray):
        if self._is_reading_active:
            try:
                # PyAudio player expects float32
                self.audio_playback_queue.put(audio_data_np_float32)
            except queue.Full:
                logger.warning("Audio playback queue is full. Skipping chunk.")
        else:
            logger.warning(
                "Audio chunk generated, but reading not active. Discarding.")

    def _on_generation_thread_finished(self):
        logger.debug(
            "AudioGenerator thread finished. Signaling player queue with None.")
        self._generation_complete_flag = True  # Mark generation as complete
        try:
            self.audio_playback_queue.put(None)  # Signal end to player
        except queue.Full:
            logger.warning(
                "Playback queue full when trying to send generation finished signal.")
           # If player queue is full here, it's problematic. Player might not get the None signal.

    def _on_playback_thread_finished(self):
        logger.debug(
            "AudioPlayerPyAudio thread finished (playback_finished signal).")
        # This signal means the player thread's run() method has exited.
        # Finalize, ensuring generation was also marked complete.
        if self._is_reading_active:  # Only finalize if we were actively reading
            self._finalize_reading_session(from_playback_finish=True)
        else:
            logger.debug(
                "Playback thread finished, but reading session was not active. No finalization needed from here.")

    def _on_tts_error(self, error_message):
        logger.error(f"TTS Generation Error: {error_message}")
        QMessageBox.critical(None, "TTS Generation Error", error_message)
        self.stop_reading(silent=True)

    def _on_playback_error(self, error_message):
        logger.error(f"Audio Playback Error: {error_message}")
        QMessageBox.critical(None, "Audio Playback Error", error_message)
        self.stop_reading(silent=True)

    def _finalize_reading_session(self, from_playback_finish=False):
        logger.debug(
            f"Finalizing reading session. From playback_finish: {from_playback_finish}, Active: {self._is_reading_active}")

        # Defensive: if not active, don't do anything unless forced by a specific path
        # from_playback_finish might mean it just became inactive
        if not self._is_reading_active and not from_playback_finish:
            logger.debug(
                "Finalize called but not active and not from playback finish. Aborting finalize.")
            return

        # Stop generator if it's still running
        if self.audio_generator_thread:
            if self.audio_generator_thread.isRunning():
                logger.debug("Finalize: Stopping AudioGenerator thread.")
                self.audio_generator_thread.stop()
                if not self.audio_generator_thread.wait(1000):  # Shorter wait
                    logger.warning(
                        "AudioGenerator did not stop gracefully. Terminating.")
                    self.audio_generator_thread.terminate()
                    self.audio_generator_thread.wait()
            self.audio_generator_thread.deleteLater()
            self.audio_generator_thread = None

        # Stop player if it's still running
        if self.audio_player_thread:
            if self.audio_player_thread.isRunning():
                logger.debug("Finalize: Stopping AudioPlayerPyAudio thread.")
                self.audio_player_thread.stop()
                if not self.audio_player_thread.wait(1000):  # Shorter wait
                    logger.warning(
                        "AudioPlayerPyAudio did not stop gracefully. Terminating.")
                    self.audio_player_thread.terminate()
                    self.audio_player_thread.wait()
            self.audio_player_thread.deleteLater()
            self.audio_player_thread = None

        # Clear queues after threads are confirmed stopped/dealt with
        for q_name, q_obj in [("TextProcessingQueue", self.text_processing_queue), ("AudioPlaybackQueue", self.audio_playback_queue)]:
            cleared_count = 0
            while not q_obj.empty():
                try:
                    q_obj.get_nowait()
                    cleared_count += 1
                except queue.Empty:
                    break
            if cleared_count > 0:
                logger.debug(f"Cleared {cleared_count} items from {q_name}.")

        self._is_reading_active = False  # Crucial: set before further UI updates
        self._generation_complete_flag = False
        self.stop_action.setEnabled(False)
        self.tray_icon.setToolTip("TTS Idle")
        logger.info("Reading session finalized.")

    def stop_reading(self, silent=False):
        logger.info("Stop reading requested.")
        if not self._is_reading_active:
            if not silent:
                self.tray_icon.showMessage(
                    "TTS", "Nothing is currently being read.", QSystemTrayIcon.Information, 1500)
            return
        self._finalize_reading_session()
        if not silent:
            self.tray_icon.showMessage(
                "TTS", "Reading stopped.", QSystemTrayIcon.Information, 1500)

    def save_to_wav(self):
        if not self.model:
            self.tray_icon.showMessage(
                "TTS Not Ready", "Model not loaded.", QSystemTrayIcon.Warning, 2500)
            return
        text = self.app.clipboard().text()
        if not text.strip():
            QMessageBox.warning(None, 'Clipboard Empty', 'Nothing to save.')
            return
        safe_prefix = re.sub(r'[^\w-]', '_', text[:20]
                             ).strip('_') or "tts_output"
        default_fn = str(Path.home()/"Documents"/f"{safe_prefix}.wav")
        fp, _ = QFileDialog.getSaveFileName(
            None, 'Save Audio As WAV', default_fn, 'WAV Files (*.wav)')
        if not fp:
            return
        self.tray_icon.showMessage(
            "Saving Audio", f"Starting save to {Path(fp).name}...", QSystemTrayIcon.Information, 2000)
        prompt = self.get_current_voice_prompt_path()
        task = SaveToWavTask(self.model, text, fp, prompt)
        task.signals.finished.connect(self._on_save_task_finished)
        task.signals.error.connect(self._on_save_task_error)
        self.thread_pool.start(task)

    def _on_save_task_finished(self, fp): self.tray_icon.showMessage(
        "Save Complete", f"Audio saved to:\n{Path(fp).name}", QSystemTrayIcon.Information, 3000)

    def _on_save_task_error(self, err): QMessageBox.critical(
        None, "Save Error", f"Failed to save audio:\n{err}")

    def quit_application(self):
        logger.info("Quit action triggered. Shutting down.")
        self.stop_reading(silent=True)
        self.thread_pool.waitForDone(1000)  # Short wait for save tasks
        if self.p_audio_instance:
            logger.info("Terminating PyAudio instance.")
            self.p_audio_instance.terminate()
        if self.tray_icon:
            self.tray_icon.hide()
            self.tray_icon.deleteLater()
        self.app.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("ClipboardTTSReaderPyAudio")
    app.setOrganizationName("YourOrgName")

    # Pre-check for CUDA only if ChatterboxTTS is intended to be used
    if CHATTERBOX_AVAILABLE and not torch.cuda.is_available():
        QMessageBox.critical(None, 'GPU Error - Pre-check',
                             'NVIDIA CUDA GPU is required but not detected for TTS.\nApplication will exit.')
        sys.exit(1)
    if CHATTERBOX_AVAILABLE:
        logger.info(
            f"CUDA is available. GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    app.setQuitOnLastWindowClosed(False)
    try:
        reader = ClipboardReader(app_instance=app)
    except Exception as e:
        logger.critical(
            f"Failed to initialize ClipboardReader: {e}", exc_info=True)
        QMessageBox.critical(None, "Application Error",
                             f"A critical error occurred during initialization:\n{e}\n\nThe application will now exit.")
        sys.exit(1)

    logger.info(
        "Application started successfully. ClipboardReader is active (PyAudio backend).")
    sys.exit(app.exec())
