import os
import re
import torch
import whisper
import queue
import argparse
import threading
import numpy as np
import speech_recognition as sr


QUERY_QUEUE = queue.Queue()


class SpeechRecognizer:
    def __init__(self, model, english, energy, pause, dynamic_energy, wake_word=None):
        """
        初始化模型所需的配置参数
        """
        self.english = english
        self.energy = energy
        self.pause = pause
        self.dynamic_energy = dynamic_energy
        self.wake_word = wake_word

        if model != "large" and english:
            model = f"{model}.en"

        self.trans_model = whisper.load_model(model)
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()

    def record_audio(self):
        """
        录制音频
        """
        r = sr.Recognizer()
        r.energy_threshold = self.energy
        r.pause_threshold = self.pause
        r.dynamic_energy_threshold = self.dynamic_energy

        with sr.Microphone(sample_rate=16000) as source:
            print("Listening...")
            while True:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                audio_data = torch.from_numpy(
                    np.frombuffer(audio.get_raw_data(), np.int16)
                    .flatten()
                    .astype(np.float32)
                    / 32768.0
                )
                self.audio_queue.put_nowait(audio_data)

    def transcribe_audio(self):
        """
        将语音转化为文本
        """
        while True:
            audio_data = self.audio_queue.get()
            if self.english:
                result = self.trans_model.transcribe(audio_data, language=self.english)
            else:
                result = self.trans_model.transcribe(audio_data)
            text = result["text"]

            if not self.wake_word:
                predicted_text = text
            else:
                if text.strip().lower().startswith(self.wake_word.strip().lower()):
                    pattern = re.compile(re.escape(self.wake_word), re.IGNORECASE)
                    predicted_text = pattern.sub("", text).strip()
                else:
                    print("You did not say the wake word, ignoring...")
                    continue

            punc = """,!()-[]{};:'"\,<>./?@#$%^&*_~"""
            self.text_queue.put_nowait(predicted_text)
            print("Got it: {}".format(predicted_text))


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="large", help="使用的whisper模型权重")
    parser.add_argument("--english", default=False, help="是否限制语言类型为英语")
    parser.add_argument("--energy", default=500, help="固定的用于检测声音的阈值")
    parser.add_argument("--pause", default=1.5, help="间隔用于检测短句")
    parser.add_argument("--dynamic_energy", default=True, help="设置动态能量阈值")
    parser.add_argument("--wake_word", default="hey", help="用于唤醒llm响应的唤醒词")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    speech_recognition = SpeechRecognizer(
        args.model,
        args.english,
        args.energy,
        args.pause,
        args.dynamic_energy,
        args.wake_word,
    )

    audio_thread = threading.Thread(target=speech_recognition.record_audio)
    trans_thread = threading.Thread(target=speech_recognition.transcribe_audio)

    audio_thread.start()
    trans_thread.start()

    audio_thread.join()
    trans_thread.join()
