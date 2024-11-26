import matplotlib.pyplot as plt
import torch.nn as nn

class LightweightVisualizer:
    def __init__(self):
        self.activations = {}
        self.model = None
        self.hooks = []

    def load(self, model):
        # Register hooks only for layers we want to visualize
        self.model = model
        self.hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(
                    module.register_forward_hook(
                        lambda m, inp, out, name=name: self.activations.update({name: out})
                    )
                )
    
    def interpret(self, spec):
        """Fast visualization without sonification"""
        plt.figure(figsize=(10, 3))
        
        # Add channel dimension if needed
        if len(spec.shape) == 3:
            spec = spec.unsqueeze(0)  # Add channel dimension [1, 128, 90] -> [1, 1, 128, 90]
        
        # Input spectrogram
        plt.subplot(1, 2, 1)
        spec_display = spec[0, 0].detach().cpu().numpy()  # Now should be [128, 90]
        plt.imshow(spec_display, aspect='auto', origin='lower')
        plt.title('Input')
        
        # Forward pass to get activations
        _ = self.model(spec)
        
        # First conv layer's activation
        first_conv = list(self.activations.values())[0]
        mean_activation = first_conv[0].mean(dim=0).detach().cpu().numpy()
        plt.subplot(1, 2, 2)
        plt.imshow(mean_activation, aspect='auto', origin='lower')
        plt.title('Conv1 Features')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

        # show the other layers
        for i, (name, activation) in enumerate(self.activations.items()):
            if i > 0:
                plt.figure(figsize=(10, 3))
                plt.imshow(activation[0].mean(dim=0).detach().cpu().numpy(), aspect='auto', origin='lower')
                plt.title(f'{name} Features')
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()