import numpy as np

class Listener:
    def __init__(self, model_architecture, model_weights):
        self.model_architecture = model_architecture
        self.model_weights = model_weights

    def listen(self, audio_buffer: np.array):
        print("listening to...")
        print(audio_buffer)
        return True