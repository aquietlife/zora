import numpy as np
import librosa

class DataAugmentation:

    def __init__(self):
        return

    def noise(self, data, noise_amt=0.035):
        #print(f"self: {type(self)}")
        #print(f"data: {type(data)}, noise_amt: {type(noise_amt)}")
        #print(f"data is self: {data is self}")

        noise_amp = noise_amt * np.random.uniform() * np.amax(data)
        noise = noise_amp * np.random.normal(size=data.shape[0])
        return data + noise

    def stretch(self, data, rate=0.8):
        return librosa.effects.time_stretch(data, rate=rate)

    def shift(self, data):
        shift_range = int(np.random.uniform(low=-5, high = 5) * 1000)
        return np.roll(data, shift_range)


    def pitch(self, data, sampling_rate, n_steps=2):
        return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=n_steps)