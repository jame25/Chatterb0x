import sys
import pyperclip
import torchaudio as ta
import torch
import pyaudio
import numpy as np
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction, QMessageBox, QStyle, QFileDialog, QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, QThread, pyqtSignal, QTimer, QObject
from chatterbox.tts import ChatterboxTTS
import threading
import traceback
from pathlib import Path
import functools
import queue
import re
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clipboard_reader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Patch torch.load to always use CPU
original_torch_load = torch.load
@functools.wraps(original_torch_load)
def patched_torch_load(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu')
    return original_torch_load(*args, **kwargs)

class AudioGenerator(QThread):
    audio_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, text_queue, voice_prompt=None):
        super().__init__()
        logger.debug("Initializing AudioGenerator")
        self.model = model
        self.text_queue = text_queue
        self.voice_prompt = voice_prompt
        self.is_running = True
        self.buffer_size = 5
        self.processing = False

    def run(self):
        try:
            logger.debug("AudioGenerator: Starting generation")
            while self.is_running:
                try:
                    text = self.text_queue.get(timeout=0.1)
                    if text is None:
                        logger.debug("AudioGenerator: Received stop signal")
                        break
                    
                    logger.debug(f"AudioGenerator: Processing text chunk: {text[:50]}...")
                    self.processing = True
                    
                    if self.voice_prompt:
                        logger.debug(f"AudioGenerator: Using voice prompt: {self.voice_prompt}")
                        wav = self.model.generate(text, audio_prompt_path=self.voice_prompt)
                    else:
                        wav = self.model.generate(text)
                    
                    audio_data = wav.numpy().astype(np.float32)
                    logger.debug(f"AudioGenerator: Generated audio data of length {len(audio_data)}")
                    
                    self.audio_ready.emit(audio_data)
                    self.processing = False
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"AudioGenerator: Error during generation: {str(e)}", exc_info=True)
                    self.processing = False
                    self.error.emit(str(e))
                    break
        except Exception as e:
            logger.error(f"AudioGenerator: Fatal error in run loop: {str(e)}", exc_info=True)
        finally:
            logger.debug("AudioGenerator: Finished generation")
            self.finished.emit()

    def stop(self):
        self.is_running = False
        # Wait for current processing to finish
        while self.processing:
            self.msleep(100)  # Sleep for 100ms
        self.text_queue.put(None)  # Signal to stop

class AudioPlayer(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, audio_queue, sample_rate):
        super().__init__()
        self.audio_queue = audio_queue
        self.base_sample_rate = sample_rate
        self.is_running = True
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.CHUNK_SIZE = 256  # Smaller chunk size for faster response

    def initialize_stream(self):
        """Initialize or reinitialize the audio stream"""
        print("AudioPlayer: Initializing stream...")  # Debug log
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None

        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.base_sample_rate,
            output=True,
            frames_per_buffer=self.CHUNK_SIZE,
            stream_callback=None,
            start=False
        )
        self.stream.start_stream()
        print("AudioPlayer: Stream initialized and started")  # Debug log

    def run(self):
        try:
            print("AudioPlayer: Starting playback...")  # Debug log
            self.initialize_stream()

            while self.is_running:
                try:
                    # Get audio data from queue with timeout
                    audio_data = self.audio_queue.get(timeout=0.1)
                    if audio_data is None:  # Signal to stop
                        print("AudioPlayer: Received stop signal")  # Debug log
                        break
                    
                    # Check if we should still be running before playing
                    if not self.is_running:
                        break
                    
                    print(f"AudioPlayer: Playing audio chunk of length {len(audio_data)}")  # Debug log
                    
                    # Play the audio in smaller chunks
                    chunk_size = self.CHUNK_SIZE * 4  # 4 chunks at a time
                    for i in range(0, len(audio_data), chunk_size):
                        if not self.is_running:
                            break
                        chunk = audio_data[i:i + chunk_size]
                        if len(chunk) > 0:
                            self.stream.write(chunk.tobytes())
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"AudioPlayer: Error during playback: {str(e)}")  # Debug log
                    self.error.emit(str(e))
                    break
        finally:
            print("AudioPlayer: Finished playback")  # Debug log
            self.stop_stream()
            self.finished.emit()

    def stop_stream(self):
        """Stop the audio stream without closing it"""
        print("AudioPlayer: Stopping stream...")  # Debug log
        if self.stream:
            try:
                self.stream.stop_stream()
            except:
                pass

    def stop(self):
        """Stop current audio playback without closing the application"""
        print("AudioPlayer: Stopping playback...")  # Debug log
        self.is_running = False
        self.stop_stream()
        
        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.audio_queue.put(None)  # Signal to stop
        print("AudioPlayer: Playback stopped")  # Debug log

    def __del__(self):
        """Clean up resources when the player is destroyed"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        if self.p:
            self.p.terminate()

class SaveToWavThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, model, text, file_path, voice_prompt=None):
        super().__init__()
        self.model = model
        self.text = text
        self.file_path = file_path
        self.voice_prompt = voice_prompt
        
    def run(self):
        try:
            logger.debug("Starting audio generation in thread")
            logger.debug(f"Text to generate: {self.text[:50]}...")
            logger.debug(f"Voice prompt: {self.voice_prompt}")
            
            # Generate audio for the entire text
            try:
                if self.voice_prompt:
                    logger.debug("Generating audio with voice prompt")
                    wav = self.model.generate(self.text, audio_prompt_path=self.voice_prompt)
                else:
                    logger.debug("Generating audio without voice prompt")
                    wav = self.model.generate(self.text)
                    
                logger.debug("Audio generation completed")
                
            except Exception as e:
                logger.error(f"Error during audio generation: {e}", exc_info=True)
                self.error.emit(f"Failed to generate audio: {str(e)}")
                return
                
            try:
                logger.debug(f"Saving audio to: {self.file_path}")
                ta.save(self.file_path, wav, self.model.sr)
                logger.info(f"Audio saved successfully to: {self.file_path}")
                self.finished.emit()
                
            except Exception as e:
                logger.error(f"Error saving audio file: {e}", exc_info=True)
                self.error.emit(f"Failed to save audio file: {str(e)}")
                
        except Exception as e:
            logger.error(f"Unexpected error in save thread: {e}", exc_info=True)
            self.error.emit(f"Unexpected error: {str(e)}")

    def save_to_wav(self):
        """Save clipboard text as a WAV file"""
        try:
            text = self.get_clipboard_text()
            if not text:
                QMessageBox.warning(None, "Warning", "Clipboard is empty!")
                return
                
            # Get save location from user
            file_path, _ = QFileDialog.getSaveFileName(
                None,
                "Save Audio File",
                str(Path.home() / "Documents" / "tts_output.wav"),
                "WAV Files (*.wav)"
            )
            
            if not file_path:  # User cancelled
                return
                
            # Get the current voice prompt
            voice_prompt = str(self.voice_prompts.get(self.current_voice)) if self.current_voice else None
            
            # Create and start the save thread
            self.save_thread = SaveToWavThread(self.model, text, file_path, voice_prompt)
            self.save_thread.finished.connect(self.on_save_finished)
            self.save_thread.error.connect(self.on_save_error)
        
            # Keep a reference to the thread
            self._save_thread_ref = self.save_thread
            
            logger.debug("Starting save thread")
            self.save_thread.start()
            
        except Exception as e:
            logger.error(f"Error in save_to_wav: {e}", exc_info=True)
            QMessageBox.critical(None, "Error", f"Failed to save audio: {e}")
            
    def on_save_finished(self):
        """Handle successful save completion"""
        try:
            logger.debug("Save operation completed successfully")
            if hasattr(self, 'save_thread'):
                self.save_thread.deleteLater()
                self.save_thread = None
            if hasattr(self, '_save_thread_ref'):
                delattr(self, '_save_thread_ref')
        except Exception as e:
            logger.error(f"Error in on_save_finished: {e}", exc_info=True)
            
    def on_save_error(self, error_msg):
        """Handle save error"""
        try:
            logger.error(f"Save operation failed: {error_msg}")
            QMessageBox.critical(None, "Error", error_msg)
            if hasattr(self, 'save_thread'):
                self.save_thread.deleteLater()
                self.save_thread = None
            if hasattr(self, '_save_thread_ref'):
                delattr(self, '_save_thread_ref')
        except Exception as e:
            logger.error(f"Error in on_save_error: {e}", exc_info=True)

class SaveDialog(QDialog):
    def __init__(self, reader, model, text, voice_prompt=None):
        super().__init__(None)
        self.model = model
        self.text = text
        self.voice_prompt = voice_prompt
        self.reader = reader
        self.setWindowTitle("Saving Audio")
        self.setModal(False)  # Make dialog non-modal
        
        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Add status label
        self.status_label = QLabel("Preparing to save audio...")
        layout.addWidget(self.status_label)
        
        # Add progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress)
        
        # Start save operation
        QTimer.singleShot(0, self.start_save)
        
    def start_save(self):
        try:
            logger.debug("Starting audio generation in save dialog")
            self.status_label.setText("Generating audio...")
            if self.voice_prompt:
                logger.debug(f"Using voice prompt: {self.voice_prompt}")
                wav = self.model.generate(self.text, audio_prompt_path=self.voice_prompt)
            else:
                logger.debug("Generating without voice prompt")
                wav = self.model.generate(self.text)
                
            logger.debug("Audio generation completed")
            self.status_label.setText("Saving to file...")
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Audio File",
                str(Path.home() / "Documents" / "tts_output.wav"),
                "WAV Files (*.wav)"
            )
            
            if file_path:
                logger.debug(f"Saving to: {file_path}")
                ta.save(file_path, wav, self.model.sr)
                self.status_label.setText(f"Audio saved successfully to:\n{file_path}")
                logger.info(f"Audio saved successfully to: {file_path}")
            else:
                logger.debug("Save cancelled by user")
                self.status_label.setText("Save cancelled")
                
        except Exception as e:
            logger.error(f"Error in save dialog: {e}", exc_info=True)
            self.status_label.setText(f"Error: {str(e)}")
            
        finally:
            # Close dialog after a short delay
            QTimer.singleShot(2000, self.close)
            
    def closeEvent(self, event):
        """Handle dialog close event"""
        logger.debug("Save dialog close event")
        # Ensure tray icon stays visible
        if hasattr(self.reader, 'tray'):
            self.reader.tray.show()
        # Process any pending events
        if hasattr(self.reader, 'app'):
            self.reader.app.processEvents()
        super().closeEvent(event)

class ClipboardReader(QObject):
    def __init__(self, app):
        try:
            logger.debug("Initializing ClipboardReader")
            super().__init__()
            self.app = app
            
            # Keep a reference to the application
            self._app_reference = app
            
            # Connect to clipboard changed signal
            self.app.clipboard().dataChanged.connect(self.on_clipboard_changed)
            
            # Store the current clipboard text
            self.current_clipboard_text = ""
            
            logger.debug("Creating system tray icon")
            self.tray = QSystemTrayIcon()
            
            # Use the new icon file
            icon = QIcon("app_icon.png")
            self.tray.setIcon(icon)
            self.tray.setVisible(True)
            
            # Keep the tray icon reference
            self._tray_reference = self.tray
            
            logger.debug("Creating main menu")
            self.menu = QMenu()
            
            # Create read action
            read_action = QAction("Read Clipboard", self.menu)
            read_action.triggered.connect(self.read_clipboard)
            self.menu.addAction(read_action)
            
            # Create save to WAV action
            save_action = QAction("Save to WAV", self.menu)
            save_action.triggered.connect(self.save_to_wav)
            self.menu.addAction(save_action)
            
            # Create stop action
            stop_action = QAction("Stop Reading", self.menu)
            stop_action.triggered.connect(self.stop_reading)
            self.menu.addAction(stop_action)
            
            self.menu.addSeparator()
            
            logger.debug("Setting up voice menu")
            self.voice_menu = QMenu("Voice", self.menu)
            self.current_voice = "female"
            self.voice_prompts_dir = Path("voice_prompts")
            self.voice_prompts = {
                "male": self.voice_prompts_dir / "male_voice.wav",
                "female": self.voice_prompts_dir / "female_voice.wav"
            }
            
            logger.debug("Creating voice prompts directory")
            self.voice_prompts_dir.mkdir(exist_ok=True)
            
            logger.debug("Initializing TTS model")
            self.initialize_model()
            
            if not self.voice_prompts["female"].exists():
                logger.debug("Creating female voice prompt")
                self.create_female_voice_prompt()
            if not self.voice_prompts["male"].exists():
                logger.debug("Creating male voice prompt")
                self.create_male_voice_prompt()
                
            male_voice = QAction("Male Voice", self.voice_menu)
            male_voice.triggered.connect(lambda: self.set_voice("male"))
            self.voice_menu.addAction(male_voice)
            
            female_voice = QAction("Female Voice", self.voice_menu)
            female_voice.triggered.connect(lambda: self.set_voice("female"))
            self.voice_menu.addAction(female_voice)
            
            self.menu.addMenu(self.voice_menu)
            self.menu.addSeparator()
            
            logger.debug("Setting up auto-read toggle")
            self.auto_read_action = QAction("Auto-Read Clipboard", self.menu)
            self.auto_read_action.setCheckable(True)
            self.auto_read_action.setChecked(False)
            self.auto_read_action.triggered.connect(self.toggle_auto_read)
            self.menu.addAction(self.auto_read_action)
            
            self.menu.addSeparator()
            
            quit_action = QAction("Quit", self.menu)
            quit_action.triggered.connect(self.app.quit)
            self.menu.addAction(quit_action)
            
            self.tray.setContextMenu(self.menu)
            
            logger.debug("Initializing queues and threads")
            self.text_queue = queue.Queue()
            self.audio_queue = queue.Queue()
            self.generator = None
            self.player = None
            
            logger.debug("Initializing clipboard monitoring")
            self.last_clipboard_text = ""
            try:
                clipboard = self.app.clipboard()
                text = clipboard.text()
                if text:
                    self.last_clipboard_text = text
                    logger.debug(f"Initial clipboard content: {repr(self.last_clipboard_text)}")
            except Exception as e:
                logger.error(f"Error initializing clipboard: {e}", exc_info=True)
                
            self.clipboard_timer = QTimer()
            self.clipboard_timer.timeout.connect(self.check_clipboard)
            
            self.is_reading = False
            self.pending_read = None
            
            logger.debug("ClipboardReader initialization complete")
            
        except Exception as e:
            logger.error(f"Error during ClipboardReader initialization: {e}", exc_info=True)
            raise
        
    def create_female_voice_prompt(self):
        """Create a sample female voice prompt using the model's default female voice"""
        try:
            # Generate a short sample with a female voice
            sample_text = "Hello, this is a sample of my voice."
            wav = self.model.generate(sample_text)
            ta.save(str(self.voice_prompts["female"]), wav, self.model.sr)
            print(f"Created female voice prompt at: {self.voice_prompts['female']}")
        except Exception as e:
            print(f"Error creating female voice prompt: {e}")
            
    def create_male_voice_prompt(self):
        """Create a sample male voice prompt using the model"""
        try:
            # Generate a short sample with a male voice
            # Note: You may need to adjust this based on how ChatterboxTTS handles male voices
            sample_text = "Hello, this is a sample of my voice."
            wav = self.model.generate(sample_text)
            ta.save(str(self.voice_prompts["male"]), wav, self.model.sr)
            print(f"Created male voice prompt at: {self.voice_prompts['male']}")
        except Exception as e:
            print(f"Error creating male voice prompt: {e}")
        
    def set_voice(self, voice_type):
        """Set the current voice type"""
        self.current_voice = voice_type
        print(f"Voice set to: {voice_type}")
        print(f"Using voice prompt: {self.voice_prompts[voice_type]}")
        
    def initialize_model(self):
        try:
            print("Initializing TTS model...")
            if torch.cuda.is_available():
                print("CUDA is available, using GPU")
                self.model = ChatterboxTTS.from_pretrained(device="cuda")
            else:
                print("CUDA is not available, using CPU")
                # Temporarily patch torch.load
                torch.load = patched_torch_load
                try:
                    self.model = ChatterboxTTS.from_pretrained(device="cpu")
                finally:
                    # Restore original torch.load
                    torch.load = original_torch_load
                
                # Show warning about CPU usage
                QMessageBox.warning(None, "GPU Not Available",
                                  "CUDA is not available. The application will run on CPU, which may be slower.\n\n"
                                  "To enable GPU acceleration:\n"
                                  "1. Install NVIDIA GPU drivers\n"
                                  "2. Install CUDA Toolkit\n"
                                  "3. Install cuDNN\n"
                                  "4. Reinstall PyTorch with CUDA support")
            print("TTS model initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize TTS model:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(None, "Error", error_msg)
            sys.exit(1)
    
    def split_text(self, text):
        # Split text into smaller chunks for faster initial response
        # Optimize chunk size for lower latency
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # For very short texts, return as single chunk for faster processing
        if len(text) <= 50:
            return [text]
            
        # For longer texts, split into smaller chunks
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_length = 50  # Reduced from 100 for faster initial response
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If this is the first chunk and it's short enough, add it immediately
            if not chunks and sentence_length <= max_chunk_length:
                chunks.append(sentence)
                continue
            
            # If adding this sentence would exceed max length, start new chunk
            if current_length + sentence_length > max_chunk_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
            
            # If we have 1 sentence or reached max length, add to chunks
            if len(current_chunk) >= 1 and current_length >= max_chunk_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        # Add any remaining sentences
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def toggle_auto_read(self, checked):
        """Toggle automatic clipboard reading"""
        try:
            logger.debug(f"Toggling auto-read: {checked}")
            if checked:
                # Update last clipboard text to current content to avoid immediate read
                current_text = self.get_clipboard_text()
                if current_text:
                    self.last_clipboard_text = current_text
                self.clipboard_timer.start(500)
                logger.debug("Auto-read enabled")
            else:
                self.clipboard_timer.stop()
                logger.debug("Auto-read disabled")
        except Exception as e:
            logger.error(f"Error toggling auto-read: {e}", exc_info=True)
            # Reset the auto-read state if there was an error
            self.auto_read_action.setChecked(False)
            self.clipboard_timer.stop()
            
    def check_clipboard(self):
        """Check if clipboard content has changed"""
        try:
            current_text = self.get_clipboard_text()
            
            if current_text and current_text.strip() != self.last_clipboard_text.strip():
                logger.debug(f"New clipboard content detected: {current_text[:50]}...")
                self.last_clipboard_text = current_text
                
                if self.is_reading:
                    logger.debug("Currently reading, queuing new text")
                    self.pending_read = current_text
                else:
                    logger.debug("Starting new read")
                    self.read_clipboard(text=current_text)
                
        except Exception as e:
            logger.error(f"Error checking clipboard: {e}", exc_info=True)
            self.last_clipboard_text = ""
            
    def get_clipboard_text(self):
        """Get text from clipboard using Qt's clipboard"""
        try:
            # First try to get the stored text
            if self.current_clipboard_text:
                return self.current_clipboard_text
                
            # If no stored text, try to get from clipboard
            clipboard = self.app.clipboard()
            if not clipboard:
                logger.error("Failed to get clipboard instance")
                return None
                
            mime_data = clipboard.mimeData()
            if not mime_data:
                logger.error("No mime data available")
                return None
                
            if mime_data.hasText():
                text = mime_data.text()
                if text and text.strip():
                    # Store the text
                    self.current_clipboard_text = text
                    logger.debug(f"New clipboard text stored: {text[:50]}...")
                    return text
                else:
                    logger.debug("Clipboard text is empty or whitespace only")
                    return None
            else:
                logger.debug("Clipboard content is not text")
                return None
                
        except Exception as e:
            logger.error(f"Fatal error getting clipboard text: {e}", exc_info=True)
            return None
            
    def read_clipboard(self, text=None):
        try:
            logger.debug("Starting read_clipboard")
            logger.debug(f"Input text parameter type: {type(text)}, value: {repr(text)}")
            logger.debug(f"Current stored text: {repr(self.current_clipboard_text)}")
            logger.debug(f"Last clipboard text: {repr(self.last_clipboard_text)}")
            
            # If text is None or False, try to get from clipboard
            if text is None or text is False:
                logger.debug("No valid text provided, getting from clipboard")
                try:
                    text = self.get_clipboard_text()
                    logger.debug(f"Got text from clipboard: {repr(text)}")
                except Exception as e:
                    logger.error(f"Error getting clipboard text: {e}", exc_info=True)
                    text = None
                
            if not text:
                logger.warning("No text in clipboard")
                QMessageBox.warning(None, "Warning", "No text found in clipboard. Please copy some text first.")
                return

            # Ensure text is a string
            if not isinstance(text, str):
                logger.warning(f"Invalid text type: {type(text)}")
                QMessageBox.warning(None, "Warning", "Invalid clipboard content. Please copy some text first.")
                return

            if not text or not text.strip():
                logger.warning("No text in clipboard or text is empty")
                QMessageBox.warning(None, "Warning", "No text found in clipboard. Please copy some text first.")
                return
            
            logger.debug(f"Got text from clipboard: {text[:50]}...")
            
            if self.is_reading:
                logger.debug("Currently reading, queuing new text")
                self.pending_read = text
                return
                
            logger.debug("Setting is_reading to True")
            self.is_reading = True
            
            try:
                logger.debug("Stopping previous reading")
                self.stop_reading()
            except Exception as e:
                logger.error(f"Error stopping previous reading: {e}", exc_info=True)
            
            try:
                chunks = self.split_text(text)
                logger.debug(f"Split text into {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error splitting text: {e}", exc_info=True)
                self.is_reading = False
                return
            
            try:
                voice_prompt = str(self.voice_prompts.get(self.current_voice)) if self.current_voice else None
                logger.debug(f"Using voice prompt: {voice_prompt}")
            except Exception as e:
                logger.error(f"Error getting voice prompt: {e}", exc_info=True)
                voice_prompt = None
            
            try:
                logger.debug("Creating new AudioGenerator and AudioPlayer")
                self.generator = AudioGenerator(self.model, self.text_queue, voice_prompt)
                self.player = AudioPlayer(self.audio_queue, self.model.sr)
            except Exception as e:
                logger.error(f"Error creating audio components: {e}", exc_info=True)
                self.is_reading = False
                return
            
            try:
                logger.debug("Connecting signals")
                self.generator.audio_ready.connect(lambda audio: self.audio_queue.put(audio))
                self.generator.error.connect(self.on_generation_error)
                self.player.error.connect(self.on_playback_error)
                self.generator.finished.connect(self.on_reading_finished)
                self.player.finished.connect(self.on_playback_finished)
            except Exception as e:
                logger.error(f"Error connecting signals: {e}", exc_info=True)
                self.is_reading = False
                return
            
            try:
                logger.debug("Starting generator and player threads")
                self.generator.start(QThread.HighPriority)
                self.player.start(QThread.HighPriority)
            except Exception as e:
                logger.error(f"Error starting threads: {e}", exc_info=True)
                self.is_reading = False
                return
            
            try:
                logger.debug("Adding text chunks to queue")
                for chunk in chunks:
                    logger.debug(f"Adding chunk to queue: {chunk[:50]}...")
                    self.text_queue.put(chunk)
                self.text_queue.put(None)
                logger.debug("Finished adding chunks to queue")
            except Exception as e:
                logger.error(f"Error adding chunks to queue: {e}", exc_info=True)
                self.is_reading = False
                return
            
        except Exception as e:
            logger.error(f"Error in read_clipboard: {e}", exc_info=True)
            self.is_reading = False
            try:
                self.clear_queues()
            except Exception as clear_error:
                logger.error(f"Error clearing queues: {clear_error}", exc_info=True)
            QMessageBox.critical(None, "Error", f"Failed to read clipboard: {e}")
    
    def on_generation_error(self, msg):
        """Handle generation errors"""
        print(f"Generation error: {msg}")
        self.is_reading = False
        QMessageBox.critical(None, "Error", f"Generation error: {msg}")
    
    def on_playback_error(self, msg):
        """Handle playback errors"""
        print(f"Playback error: {msg}")
        self.is_reading = False
        QMessageBox.critical(None, "Error", f"Playback error: {msg}")
    
    def clear_queues(self):
        """Clear both text and audio queues"""
        try:
            logger.debug("Clearing queues")
            while not self.text_queue.empty():
                try:
                    item = self.text_queue.get_nowait()
                    logger.debug(f"Cleared text queue item: {str(item)[:50] if item else 'None'}")
                except queue.Empty:
                    break
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            logger.debug("Queues cleared")
        except Exception as e:
            logger.error(f"Error clearing queues: {e}", exc_info=True)

    def check_queue_status(self):
        """Debug method to check queue status"""
        print(f"Text queue size: {self.text_queue.qsize()}")
        print(f"Audio queue size: {self.audio_queue.qsize()}")
        print(f"Is reading: {self.is_reading}")
        print(f"Pending read: {self.pending_read[:50] if self.pending_read else 'None'}")
            
    def stop_reading(self):
        """Stop current reading without closing the application"""
        print("Stopping reading...")  # Debug log
        # Reset the reading flag
        self.is_reading = False
        
        # Stop player thread first to immediately stop audio
        if hasattr(self, 'player') and self.player:
            try:
                self.player.finished.disconnect()
            except:
                pass
            self.player.stop()
            self.player.wait()
            self.player = None
            
        # Then stop generator thread
        if hasattr(self, 'generator') and self.generator:
            try:
                self.generator.finished.disconnect()
            except:
                pass
            self.generator.stop()
            self.generator.wait()
            self.generator = None
            
        # Clear queues after threads are stopped
        self.clear_queues()
        print("Reading stopped")  # Debug log
    
    def on_reading_finished(self):
        """Called when audio generation is finished"""
        print("Audio generation finished")
        # Don't reset is_reading here - wait for playback to finish
        
    def on_playback_finished(self):
        """Called when audio playback is finished"""
        print("Audio playback finished")
        # Save pending read before resetting state
        pending = self.pending_read
        self.pending_read = None
        
        # Reset reading state
        self.is_reading = False
        
        # Process pending read if exists
        if pending:
            print(f"Processing pending read: {pending[:50]}...")
            # Read the pending text immediately
            self.read_clipboard(text=pending)
        else:
            # Only update last clipboard text if there's no pending read
            # This prevents re-reading the same content
            self.last_clipboard_text = pyperclip.paste()
    
    def save_to_wav(self):
        """Save clipboard text as a WAV file"""
        logger.debug("=== Starting save_to_wav ===")
        try:
            text = self.get_clipboard_text()
            logger.debug(f"Got clipboard text: {text[:50]}...")
            if not text:
                logger.warning("Clipboard is empty")
                QMessageBox.warning(None, "Warning", "Clipboard is empty!")
                return
            
            # Get the current voice prompt
            voice_prompt = str(self.voice_prompts.get(self.current_voice)) if self.current_voice else None
            logger.debug(f"Using voice prompt: {voice_prompt}")
            
            # Create and show save dialog
            logger.debug("Creating save dialog")
            self.save_dialog = SaveDialog(self, self.model, text, voice_prompt)
            self.save_dialog.finished.connect(self.on_save_dialog_finished)
            self.save_dialog.show()
            logger.debug("Save dialog shown")
            
            # Process any pending events
            self.app.processEvents()
            
        except Exception as e:
            logger.error(f"Error in save_to_wav: {e}", exc_info=True)
            QMessageBox.critical(None, "Error", f"Failed to save audio: {e}")
        finally:
            logger.debug("=== Finished save_to_wav ===")
            
    def on_save_dialog_finished(self):
        """Clean up when save dialog is closed"""
        logger.debug("Save dialog finished")
        if hasattr(self, 'save_dialog'):
            self.save_dialog.deleteLater()
            delattr(self, 'save_dialog')
        # Ensure tray icon stays visible
        if hasattr(self, 'tray'):
            self.tray.show()
        # Process any pending events
        if hasattr(self, 'app'):
            self.app.processEvents()
    
    def run(self):
        """Run the application"""
        logger.debug("=== Starting application run ===")
        try:
            # Keep a reference to self to prevent garbage collection
            self._self_reference = self
            logger.debug("Stored self reference for run")
            
            logger.debug("Entering application event loop")
            result = self.app.exec_()
            logger.debug(f"Application event loop exited with result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in run: {e}", exc_info=True)
            return 1
        finally:
            logger.debug("Cleaning up run")
            if hasattr(self, '_self_reference'):
                delattr(self, '_self_reference')
            logger.debug("=== Finished application run ===")

    def on_clipboard_changed(self):
        """Handle clipboard changes"""
        try:
            logger.debug("Clipboard changed signal received")
            logger.debug(f"Current stored text before change: {repr(self.current_clipboard_text)}")
            # Process immediately to ensure we capture the text
            self._process_clipboard_change()
        except Exception as e:
            logger.error(f"Error in clipboard change handler: {e}", exc_info=True)

    def _process_clipboard_change(self):
        """Process clipboard change"""
        try:
            logger.debug("Processing clipboard change")
            clipboard = self.app.clipboard()
            if not clipboard:
                logger.error("No clipboard instance available")
                return
                
            logger.debug("Got clipboard instance, checking mime data")
            mime_data = clipboard.mimeData()
            if not mime_data:
                logger.error("No mime data available")
                return
                
            logger.debug("Got mime data, checking for text")
            if mime_data.hasText():
                text = mime_data.text()
                logger.debug(f"New clipboard text: {repr(text)}")
                if text and text.strip():
                    # Store the text
                    self.current_clipboard_text = text
                    self.last_clipboard_text = text
                    logger.debug(f"Stored clipboard text: {repr(self.current_clipboard_text)}")
                    # Only auto-read if auto-read mode is enabled
                    if self.auto_read_action.isChecked():
                        if not self.is_reading:
                            self.read_clipboard(text=text)  # Pass the actual text
                        else:
                            self.pending_read = text
            else:
                logger.debug("Clipboard content is not text")
                
        except Exception as e:
            logger.error(f"Error processing clipboard change: {e}", exc_info=True)

if __name__ == '__main__':
    try:
        logger.info("=== Starting application ===")
        app = QApplication(sys.argv)
        
        try:
            logger.debug("Creating ClipboardReader")
            reader = ClipboardReader(app)
            
            # Create circular references to prevent garbage collection
            app._reader = reader
            reader._app = app
            
            # Keep a reference to the main window
            app._main_window = reader
            
            # Prevent application from exiting
            app.setQuitOnLastWindowClosed(False)
            
            logger.info("Entering application event loop")
            result = app.exec_()
            logger.info(f"Application event loop exited with result: {result}")
            
            # Only exit if the result is not 0 (normal exit)
            if result != 0:
                sys.exit(result)
                
        except Exception as e:
            logger.critical(f"Error in application: {e}", exc_info=True)
            sys.exit(1)
            
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("=== Application shutdown complete ===")
