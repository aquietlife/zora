import torch as t
import pytest
from zoraspeech.architectures.speech_transformer.speech_transformer import (
    SpeechTransformer, 
    Attention, 
    Config, 
    CharacterVocabulary,
    CharacterEmbedding
    )

def test_speech_transformer_forward_pass():
    # TODO Modify this to match actual tensor size of (batch_size, time_steps, frequency_bins)
    model = SpeechTransformer()
    
    batches = 2
    time_steps = 100
    freq_bins = model.cfg.n_freq_bins

    input_tensor = t.randn(batches, time_steps, freq_bins)
    output = model(input_tensor)

    assert output.shape == (batches, time_steps // 4, model.cfg.d_model) #TODO update this as we continue to build out speech transformer

def test_positional_encoder():
    model = SpeechTransformer()
    cfg = Config()
    batches = 2
    time_steps = 100
    d_model = cfg.d_model
    
    input_tensor = t.randn(batches, time_steps, d_model)
    output_tensor = model.positional_encoder(input_tensor)

    assert output_tensor.shape == (batches, time_steps, d_model)

    assert not t.allclose(output_tensor[:, 0, :], output_tensor[:, 1, :])

    assert t.allclose(output_tensor[0, 0, :] - input_tensor[0, 0, :],
                      output_tensor[1, 0, :] - input_tensor[1, 0, :])

def test_attention_output_shape():

    cfg = Config()
    batch_size = 2
    seq_len = 4
    d_model = cfg.d_model

    input = t.ones(batch_size, seq_len, d_model).to(cfg.device)

    attention = Attention(cfg).to(cfg.device)

    assert attention.forward(input).shape == (batch_size, seq_len, d_model)

def test_character_vocabulary_encode_decode():
    cfg = Config()

    cv = CharacterVocabulary(cfg)

    str = "speech transformer"

    encoded_string = cv.encode(str)

    #print(encoded_string)

    decoded_string = cv.decode(encoded_string)

    #print(decoded_string)

    assert str == decoded_string

    assert len(encoded_string) == cfg.max_seq_length

    if cfg.include_sos_eos_tokens:
        assert encoded_string[0] == cfg.sos_idx
        assert encoded_string[1:].index(cfg.eos_idx) > 0 # EOS should be present and somewhere after SOS

    test_str_with_unknown_chars = "a quiet life 静かな生活"
    encoded_unknown_chars = cv.encode(test_str_with_unknown_chars)
    decoded_string = cv.decode(encoded_unknown_chars)
    assert decoded_string == "a quiet life "

    # test empty string
    with pytest.raises(ValueError):
        cv.encode("")

    # test padding
    short_str = "hi"
    encoded = cv.encode(short_str)
    assert len(encoded) == cfg.max_seq_length
    assert encoded[-1] == 0

    # test max length truncation
    long_str = "a" * (cfg.max_seq_length + 10)
    encoded = cv.encode(long_str)
    assert len(encoded) == cfg.max_seq_length

def test_character_embedding():

    cfg = Config()
    batch_size = 2
    seq_len = 4
    d_model = cfg.d_model

    input = t.ones(batch_size, seq_len, dtype=t.long).to(cfg.device)

    ce = CharacterEmbedding(cfg).to(cfg.device)

    assert ce.forward(input).shape == (batch_size, seq_len, d_model)