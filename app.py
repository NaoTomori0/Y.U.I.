import sys
import time
import json
import os
import warnings
import re
import torch
import speech_recognition as sr
import sounddevice as sd
import whisper
import ollama
import subprocess


def prepare_model():
    print("Создание модели Ollama...")
    try:

        subprocess.run(["ollama", "create", "yui", "-f", "Modelfile"], check=True)
        print("Модель 'yui' успешно создана.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при создании модели: {e}")
        sys.exit(1)


prepare_model()

print("Теперь работает основной код Python")

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QLabel,
    QFrame,
)
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from datetime import datetime


def get_current_timestamp():
    return datetime.now().strftime("%d.%m.%Y %H:%M")


warnings.filterwarnings("ignore")

MODEL_NAME = "yui"
WHISPER_SIZE = "base"
VOICE_URL = "https://models.silero.ai/models/tts/ru/v3_ru.pt"
HISTORY_FILE = "chat_history.json"
VOICE_PATH = "model.pt"

# ========== Загрузка моделей ==========
device = torch.device("cpu")
torch.set_num_threads(4)

if not os.path.isfile(VOICE_PATH):
    print("[Скачиваю голосовую модель Silero...]")
    torch.hub.download_url_to_file(VOICE_URL, VOICE_PATH)
    print("[Готово]")

try:
    model_tts = torch.package.PackageImporter(VOICE_PATH).load_pickle(
        "tts_models", "model"
    )
except Exception:
    model_tts = torch.load(VOICE_PATH, map_location=device)
model_tts.to(device)

print("[Загружаю Whisper...]")
whisper_model = whisper.load_model(WHISPER_SIZE)


def build_time_summary(history, max_entries=20):
    if not history:
        return "Диалог только начался."
    summary = "Последние сообщения:\n"
    for msg in history[-max_entries:]:
        role = "Пользователь" if msg["role"] == "user" else "Ассистент"
        ts = msg.get("timestamp", "неизвестно")
        # Можно добавить и сам текст сообщения (кратко), но только если он короткий
        content_preview = (
            msg["content"][:30] + "…" if len(msg["content"]) > 30 else msg["content"]
        )
        summary += f"- {role} ({ts}): «{content_preview}»\n"
    return summary.strip()


def load_history():
    """Загружает историю диалога из JSON."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                filtered = [msg for msg in data if msg.get("content", "").strip()]
                if len(filtered) != len(data):
                    print(f"[Удалено {len(data)-len(filtered)} пустых сообщений]")
                return filtered
        except Exception as e:
            print(f"[Ошибка загрузки истории]: {e}")
    return []


def save_history(messages):
    """Сохраняет историю диалога в JSON."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Ошибка сохранения истории]: {e}")


def clean_response(text):
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")

    text = re.sub(r"\s+", " ", text)
    markers = ["Нао:", "Ты:", "Yui:", "\nНао:", "\nТы:", "\nYui:"]
    for marker in markers:
        if marker in text:
            text = text.split(marker)[0]
    if text.startswith("Yui:"):
        text = text[6:].strip()
    return text.strip()


# ========== Потоки ==========
class VoiceRecognitionWorker(QThread):
    text_recognized = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def run(self):
        r = sr.Recognizer()
        r.pause_threshold = 2.0
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=1.0)
                self.error_occurred.emit("🎤 Слушаю...")
                audio = r.listen(source, timeout=10, phrase_time_limit=30)
                with open("temp.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                result = whisper_model.transcribe("temp.wav", language="ru", fp16=False)
                text = result["text"].strip()
                if os.path.exists("temp.wav"):
                    os.remove("temp.wav")
                bad_phrases = [
                    "продолжение следует",
                    "спасибо за просмотр",
                    "субтитры",
                    "текст предоставлен",
                ]
                if len(text) < 2 or any(p in text.lower() for p in bad_phrases):
                    self.error_occurred.emit("❌ Не распознано")
                else:
                    self.text_recognized.emit(text)
        except Exception as e:
            self.error_occurred.emit(f"Ошибка: {e}")


class TTSWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        try:
            clean = (
                self.text.replace("*", "")
                .replace('"', "")
                .replace("(", "")
                .replace(")", "")
            )
            if not clean.strip():
                return
            audio = model_tts.apply_tts(
                text=clean, speaker="kseniya", sample_rate=48000
            )
            sd.play(audio.numpy(), 48000)
            sd.wait()
        except Exception as e:
            print(f"[Ошибка озвучки]: {e}")
        finally:
            self.finished.emit()


def build_system_prompt(history, now_str):
    hour = datetime.now().hour
    if 5 <= hour < 12:
        greeting = "утро"
    elif 12 <= hour < 18:
        greeting = "день"
    elif 18 <= hour < 23:
        greeting = "вечер"
    else:
        greeting = "ночь"

    time_summary = build_time_summary(history, max_entries=3)  # сократим до 3

    return (
        f"Сейчас {now_str} ({greeting}).\n"
        f"{time_summary}\n\n"
        "для естественного общения (например, «ты спрашивал вчера вечером»). "
        "Не вставляй в ответы служебные метки вроде [дата] или <|im_end|>.\n"
        "Правила прощания:\n"
        "- Если пользователь говорит 'спокойной ночи', 'доброй ночи' и т.п., "
        "ты отвечаешь 'Сладких снов, Нао' или 'Доброй ночи' (не используй 'до свидания').\n"
        "- Не начинай прощание фразой 'Добрый вечер', если уже ночь.\n"
        "- Не используй формальные прощания вроде 'до свидания' без явной просьбы.\n"
        "Не повторяй шаблонные фразы из предыдущих ответов."
    )


class OllamaWorker(QThread):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, prompt, history):
        super().__init__()
        self.prompt = prompt
        self.history = history

    def run(self):
        try:
            now_str = get_current_timestamp()
            user_msg = {
                "role": "user",
                "content": self.prompt,
                "timestamp": now_str,
            }
            self.history.append(user_msg)
            save_history(self.history)

            messages_for_ollama = [
                {"role": msg["role"], "content": msg["content"]} for msg in self.history
            ]
            time_summary = build_time_summary(self.history, max_entries=15)
            system_msg = {
                "role": "system",
                "content": build_system_prompt(self.history, now_str),
            }
            messages_for_ollama.insert(0, system_msg)

            stream = ollama.chat(
                model=MODEL_NAME, messages=messages_for_ollama, stream=True
            )
            full_response = ""
            for chunk in stream:
                content = chunk["message"]["content"]
                full_response += content
                self.chunk_received.emit(content)

            full_response = clean_response(full_response)

            assistant_msg = {
                "role": "assistant",
                "content": full_response,
                "timestamp": get_current_timestamp(),
            }

            self.history.append(assistant_msg)
            save_history(self.history)
            self.finished.emit(full_response)
        except Exception as e:
            self.finished.emit(f"Ошибка: {e}")


# ========== GUI ==========
class ChatBubble(QFrame):
    def __init__(self, text, is_user=True, timestamp=""):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(0)

        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setMaximumWidth(450)
        self.label.setMinimumWidth(450)

        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.label.setMinimumSize(self.label.sizeHint() * 1.45)

        # self.label.setSizePolicy(
        #     QtWidgets.QSizePolicy.Policy.Expanding,
        #     QtWidgets.QSizePolicy.Policy.Preferred,
        # )

        self.label.setStyleSheet(
            """
            color: white; 
            border: none; 
            font-size: 14px;
            background: transparent;
            padding: 2px; 
            """
        )

        layout.addWidget(self.label)
        if timestamp:
            self.time_label = QLabel(timestamp)
            self.time_label.setStyleSheet(
                "color: #aaa; font-size: 10px; margin-top: 5px;"
            )
            self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            layout.addWidget(self.time_label)
        color = "#2b5278" if is_user else "#32353b"
        radius = "15px 15px 2px 15px" if is_user else "15px 15px 15px 2px"

        self.setStyleSheet(
            f"""
            background-color: {color};
            border-radius: {radius};
            margin: 5px;
        """
        )

    def set_text(self, text):
        self.label.setText(text)


class YuiApp(QWidget):
    def __init__(self):
        super().__init__()
        self.history = load_history()
        self.tts_active = False
        self.current_response_bubble = None
        self.init_ui()
        self.restore_history_to_chat()

    def init_ui(self):
        self.setWindowTitle("Юи")
        self.resize(450, 600)
        self.setStyleSheet("background-color: #1e1f22;")

        self.main_layout = QVBoxLayout(self)

        # Область чата
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background: transparent;")

        self.container = QWidget()
        self.chat_layout = QVBoxLayout(self.container)
        self.chat_layout.addStretch()
        self.scroll.setWidget(self.container)
        self.main_layout.addWidget(self.scroll)

        # Статус
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 12px; padding: 2px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        # Панель ввода
        input_box = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Напиши мне...")
        self.input_field.setStyleSheet(
            """
            QLineEdit { 
                background: #2b2d31; color: white; border-radius: 15px; 
                padding: 10px; border: 1px solid #3f4147;
            }
        """
        )
        self.input_field.returnPressed.connect(self.handle_send)

        self.btn_send = QPushButton("➤")
        self.btn_send.setFixedSize(40, 40)
        self.btn_send.setStyleSheet(
            "background: #5865f2; color: white; border-radius: 20px; font-size: 18px;"
        )
        self.btn_send.clicked.connect(self.handle_send)

        self.btn_voice = QPushButton("🎤")
        self.btn_voice.setFixedSize(40, 40)
        self.btn_voice.setStyleSheet(
            "background: #3f4147; color: white; border-radius: 20px; font-size: 16px;"
        )
        self.btn_voice.clicked.connect(self.start_voice_input)

        input_box.addWidget(self.input_field)
        input_box.addWidget(self.btn_send)
        input_box.addWidget(self.btn_voice)
        self.main_layout.addLayout(input_box)

        self.status_timer = QTimer()
        self.status_timer.setSingleShot(True)
        self.status_timer.timeout.connect(lambda: self.status_label.setText(""))

    def scroll_to_bottom(self):
        """Прокручивает область прокрутки в самый низ."""
        scrollbar = self.scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QApplication.processEvents()  # гарантируем применение геометрии

    def restore_history_to_chat(self):
        for msg in self.history:
            timp = msg["timestamp"] if "timestamp" in msg else ""
            if msg["role"] == "user":
                self.add_bubble(msg["content"], is_user=True, timestamp=timp)
            elif msg["role"] == "assistant":
                self.add_bubble(msg["content"], is_user=False, timestamp=timp)

    def add_bubble(self, text, is_user, timestamp=""):
        bubble = ChatBubble(text, is_user, timestamp)
        align = Qt.AlignmentFlag.AlignRight if is_user else Qt.AlignmentFlag.AlignLeft
        self.chat_layout.insertWidget(
            self.chat_layout.count() - 1, bubble, alignment=align
        )
        # Отложенная прокрутка после того, как layout обновится
        QTimer.singleShot(0, self.scroll_to_bottom)
        return bubble

    def handle_send(self):
        text = self.input_field.text().strip()
        if not text:
            return
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.btn_voice.setEnabled(False)

        self.add_bubble(text, True, timestamp=get_current_timestamp())
        self.scroll_to_bottom()

        self.current_response_bubble = self.add_bubble(
            "", is_user=False, timestamp=get_current_timestamp()
        )

        self.worker = OllamaWorker(text, self.history)
        self.worker.chunk_received.connect(self.on_chunk_received)
        self.worker.finished.connect(self.on_response_finished)
        self.worker.start()

    def on_chunk_received(self, chunk):
        if self.current_response_bubble:
            current_text = self.current_response_bubble.label.text()
            self.current_response_bubble.set_text(current_text + chunk)
            QTimer.singleShot(0, self.scroll_to_bottom)

    def on_response_finished(self, final_text):
        if self.current_response_bubble:
            self.current_response_bubble.set_text(final_text)

        self.input_field.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.btn_voice.setEnabled(True)
        self.input_field.setFocus()

        # Раскомментируйте для озвучки
        # self.tts_worker = TTSWorker(final_text)
        # self.tts_worker.finished.connect(lambda: setattr(self, "tts_active", False))
        # self.tts_active = True
        # self.tts_worker.start()

        self.current_response_bubble = None

    def start_voice_input(self):
        self.status_label.setText("🎤 Слушаю...")
        self.input_field.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.btn_voice.setEnabled(False)

        self.voice_worker = VoiceRecognitionWorker()
        self.voice_worker.text_recognized.connect(self.on_voice_recognized)
        self.voice_worker.error_occurred.connect(self.on_voice_error)
        self.voice_worker.start()

    def on_voice_recognized(self, text):
        self.status_label.setText("")
        self.input_field.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.btn_voice.setEnabled(True)
        self.input_field.setText(text)
        self.handle_send()

    def on_voice_error(self, msg):
        self.status_label.setText(msg)
        self.status_timer.start(3000)
        self.input_field.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.btn_voice.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YuiApp()
    window.show()
    sys.exit(app.exec())
