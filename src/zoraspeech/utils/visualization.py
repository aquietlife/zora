import numpy as np
import torch as t
import matplotlib.pyplot as plt

def plot_waveform(audio, sample_rate):
    audio = audio.numpy()

    num_channels, num_frames = audio.shape

    time_axis = t.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    
    for c in range(num_channels):
        axes[c].plot(time_axis, audio[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
    figure.suptitle("audio")
    plt.show()

def plot_specgram(audio, sample_rate):
    audio = audio.numpy()

    num_channels, num_frames = audio.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    
    for c in range(num_channels):
        axes[c].specgram(audio[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
    figure.suptitle("audio")
    plt.show()