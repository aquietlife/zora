#mel_freq_bins = 128
#time_steps = 90
#longest_audio_file_length = 17916

import pyaudio
import numpy as np
import soundfile as sf
import torch.nn as nn

import torchaudio
import matplotlib.pyplot as plt
from IPython.display import display, Audio
import torch as t

from zoraspeech.listener import Listener
from zoraspeech.architectures.cnn.cnn import ConvModel
from zoraspeech.interpreters.cnn.layer_visualizations import LightweightVisualizer

mel_freq_bins = 128
time_steps = 89
longest_audio_file_length = 17647

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050

p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    )

onset_threshold = 0.025
offset_threshold = 0.005

recorded_audio = []
recording_count = 0

print(" type ctrl+c to quit")

listener = Listener(
    model_architecture=ConvModel(),
    model_weights='/Users/jo/Documents/zora/src/zoraspeech/weights/audrey_model_weights_2024-10-26.pth',
    interpreter=LightweightVisualizer()
)

listener.load()

try:
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            rms = np.sqrt(audio_data ** 2).mean()

            if rms > onset_threshold:
                recording_count += 1
                print(f"recording (count: {recording_count})")
                recorded_audio.append(audio_data)

            elif rms < offset_threshold:
                if len(recorded_audio) > 0: # at least 1 second of audio
                    print("processing audio")
                    print("listening")

                    # concatenate all the recorded audio
                    recorded_audio = np.concatenate(recorded_audio)

                    # pad the audio to the longest audio file length
                    current_size = len(recorded_audio)
                    pad_size = longest_audio_file_length - current_size
                    left_pad = pad_size // 2
                    right_pad = pad_size - left_pad
                    padded_audio = np.pad(recorded_audio, (left_pad, right_pad), mode='constant')
                    
                    audio = t.tensor(np.array([padded_audio]))

                    # create spectrogram
                    spec = torchaudio.transforms.MelSpectrogram()(audio)

                    listener.listen(spec)
                recorded_audio = [] # reset recorded audio

        except KeyboardInterrupt:
            print(f"\nTotal recordings: {recording_count}")
            print("\nStopping stream...")
            stream.stop_stream()
            stream.close()
            p.terminate()

except KeyboardInterrupt:
    print(f"\nTotal recordings: {recording_count}")
    audio_queue.put(None)  # Signal the processing thread to stop
    processing_thread.join()
    print("\nStopping stream...")
    stream.stop_stream()
    stream.close()
    p.terminate()