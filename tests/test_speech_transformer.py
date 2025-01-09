import torch as t
from zoraspeech.architectures.speech_transformer.speech_transformer import SpeechTransformer


def test_speech_transformer_forward_pass():
    # TODO Modify this to match actual tensor size of (batch_size, time_steps, frequency_bins)
    model = SpeechTransformer()
    input_tensor = t.empty(32, 100, 80)
    output = model(input_tensor)
    print(output)

    assert model(input_tensor).shape == (32, 100, 80)
