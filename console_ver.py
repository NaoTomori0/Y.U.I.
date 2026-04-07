import ollama
import threading
import time
import whisper
import os
import warnings
import torch
import simpleaudio as sa
import speech_recognition as sr
import json
import sounddevice as sd


warnings.filterwarnings("ignore")

MODEL_NAME = "yui"
WHISPER_SIZE = "base"  # tiny (быстро) ну или можно base (точнее) medium
VOICE_URL = "https://models.silero.ai/models/tts/ru/v3_ru.pt"
HISTORY_FILE = "chat_history.json"
VOICE_PATH = "model.pt"

device = torch.device("cpu")
torch.set_num_threads(4)

try:
    model_tts = torch.package.PackageImporter(VOICE_PATH).load_pickle(
        "tts_models", "model"
    )
except Exception:
    model_tts = torch.load(VOICE_PATH, map_location=device)
model_tts.to(device)

print("[Загружаю слух (Whisper)...]")
whisper_model = whisper.load_model(WHISPER_SIZE)

if not os.path.isfile(VOICE_PATH):
    print("[Скачиваю голосовую модель Silero...]")
    torch.hub.download_url_to_file(VOICE_URL, VOICE_PATH)
    print("[Готово]")


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                filtered = [msg for msg in data if msg.get("content", "").strip()]
                if len(filtered) != len(data):
                    print(
                        f"[Удалено {len(data)-len(filtered)} пустых сообщений из истории]"
                    )
                return filtered
        except Exception as e:
            print(f"[Ошибка загрузки истории]: {e}")
    return []


def listen_to_me():
    r = sr.Recognizer()

    r.pause_threshold = 2.0
    # r.energy_threshold = 300
    # ------------------------------------------

    with sr.Microphone() as source:
        print("\n[Слушаю... говори спокойно]")
        r.adjust_for_ambient_noise(source, duration=1.0)

        try:
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
                return None

            print(f"Ты: {text}")
            return text
        except Exception as e:
            # print(f"Ошибка записи: {e}")
            return None


messages = load_history()
if not messages:
    print("[Новая сессия, история пуста]")
else:
    print(f"[Загружено {len(messages)} сообщений из истории]")

last_chat_time = time.time()


def save_history():
    """Сохраняет историю диалога в JSON-файл."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Ошибка сохранения истории]: {e}")


def speak(text):
    try:
        clean_text = (
            text.replace("*", "").replace('"', "").replace("(", "").replace(")", "")
        )
        if not clean_text.strip():
            return

        audio = model_tts.apply_tts(
            text=clean_text, speaker="kseniya", sample_rate=48000
        )

        sd.play(audio.numpy(), 48000)
        # sd.wait()
    except Exception as e:
        print(f"\n[Ошибка озвучки]: {e}")


def get_yui_response(prompt, is_auto=False):
    global last_chat_time
    last_chat_time = time.time()
    if is_auto:
        messages.append(
            {
                "role": "user",
                "content": "(Ты молчала долго. Скажи что-нибудь хаотичное или подразни Нао)",
            }
        )
    else:
        messages.append({"role": "user", "content": prompt})
    save_history()
    try:
        # print(f"DEBUG: Отправляю в Ollama {len(messages)} сообщений.")

        stream = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)

        print("Yui: ", end="", flush=True)
        full_response = ""

        for chunk in stream:
            content = chunk["message"]["content"]
            print(content, end="", flush=True)
            full_response += content
        print()

        full_response = clean_response(full_response)

        messages.append({"role": "assistant", "content": full_response})
        save_history()
        # speak(full_response)

        last_chat_time = time.time()

    except Exception as e:
        print(f"\n[Ошибка Ollama]: {e}")


def auto_talk_loop():
    while True:
        time.sleep(10)
        if time.time() - last_chat_time > 90:
            print()
            get_yui_response("", is_auto=True)


def clean_response(text):
    """Удаляет из ответа модели реплики за пользователя и лишние метки."""
    markers = ["Нао:", "Ты:", "Yui:", "\nНао:", "\nТы:", "\nYui:"]
    for marker in markers:
        if marker in text:
            text = text.split(marker)[0]
    if text.startswith("Yui:"):
        text = text[6:].strip()
    return text.strip()


# threading.Thread(target=auto_talk_loop, daemon=True).start()

print(f"\n---------------------------------------------")
print("1. Просто нажми ENTER, чтобы поговорить голосом.")
print("2. Либо введи текст вручную.")
print("3. Напиши 'exit', чтобы выйти.\n")

try:
    while True:
        user_input = input("Ты: ")

        if user_input.lower() in ["exit", "выход"]:
            break

        if not user_input.strip():
            user_input = listen_to_me()

        if user_input:
            get_yui_response(user_input)
except KeyboardInterrupt:
    print("\nПрощай, Nao!")
