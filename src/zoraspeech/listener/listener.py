import numpy as np
import torch as t

class Listener:
    def __init__(self, model_architecture, model_weights, interpreter):
        self.model_architecture = model_architecture
        self.model_weights = model_weights
        self.interpreter = interpreter

    def load(self):
        # get our device
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # load the model
        self.model_architecture.load_state_dict(t.load(self.model_weights, map_location=device))

        # load the interpreter
        self.interpreter.load(self.model_architecture)

    def listen(self, spec):
        # Pass spec into model
        outputs = self.model_architecture(spec)
        prediction = str(outputs.argmax().item())
        print("Model prediction:", prediction)

    def interpret(self, spec):
        self.interpreter.interpret(spec)