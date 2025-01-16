import torch as t
from zoraspeech.architectures.speech_transformer.speech_transformer import SpeechTransformer, Attention, Config

'''
def test_speech_transformer_forward_pass():
    # TODO Modify this to match actual tensor size of (batch_size, time_steps, frequency_bins)
    model = SpeechTransformer()
    input_tensor = t.empty(32, 100, 80)
    output = model(input_tensor)
    print(output)

    assert model(input_tensor).shape == (32, 100, 80)
'''

def test_attention_output_shape():

    cfg = Config()
    batch_size = 2
    seq_len = 4
    d_model = cfg.d_model

    input = t.ones(batch_size, seq_len, d_model).to(cfg.device)

    attention = Attention(cfg).to(cfg.device)

    assert attention.forward(input).shape == (batch_size, seq_len, d_model)

