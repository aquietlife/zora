import torch as t
from zoraspeech.architectures.speech_transformer.speech_transformer import SpeechTransformer, Attention, Config


def test_speech_transformer_forward_pass():
    # TODO Modify this to match actual tensor size of (batch_size, time_steps, frequency_bins)
    model = SpeechTransformer()
    
    batches = 2
    time_steps = 100
    freq_bins = model.cfg.n_freq_bins

    input_tensor = t.randn(batches, time_steps, freq_bins)
    output = model(input_tensor)

    assert output.shape == (batches, time_steps // 4, model.cfg.d_model) #TODO update this as we continue to build out speech transformer

def test_attention_output_shape():

    cfg = Config()
    batch_size = 2
    seq_len = 4
    d_model = cfg.d_model

    input = t.ones(batch_size, seq_len, d_model).to(cfg.device)

    attention = Attention(cfg).to(cfg.device)

    assert attention.forward(input).shape == (batch_size, seq_len, d_model)