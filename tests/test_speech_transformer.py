import einops
import torch as t
import pytest
from zoraspeech.architectures.speech_transformer.speech_transformer import (
    SpeechTransformer, 
    Attention, 
    Config, 
    CharacterEmbedding
    )

from zoraspeech.datasets.speech_transformer.speech_transformer_dataset import (
    CharacterVocabulary,
    CommonVoiceDataset,
    create_collate_fn
)

from zoraspeech.learners.speech_transformer.learner import (
    SpeechTransformerLearner
)

import torchaudio

@pytest.fixture
def device():
    return t.device("cuda" if t.cuda.is_available() else "cpu")

@pytest.fixture

def model(device):
    cfg = Config()
    model = SpeechTransformer(cfg).to(device)
    return model

@pytest.fixture
def cfg():
    return Config()

def test_speech_transformer(model, cfg):

    cv = CharacterVocabulary(cfg)
    
    batches = 2
    time_steps = 100
    freq_bins = model.cfg.n_freq_bins

    text_input = "listening is a practice of freedom"
    encoded_text_input = cv.encode(text_input)

    assert len(encoded_text_input) == model.cfg.max_seq_length

    encoded_text_tensor = t.tensor(encoded_text_input, dtype=t.long)

    encoded_text_tensor = einops.repeat(encoded_text_tensor, "seq_length -> b seq_length", b=batches).to(cfg.device)

    spectrograms = t.randn(batches, time_steps, freq_bins).to(cfg.device)

    output = model(spectrograms, encoded_text_tensor)

    assert output.shape == (batches, model.cfg.max_seq_length, model.cfg.vocab_size)

def test_positional_encoder(model, cfg):
    batches = 2
    time_steps = 100
    d_model = cfg.d_model
    
    input_tensor = t.randn(batches, time_steps, d_model)
    output_tensor = model.encoder.positional_encoder(input_tensor)

    assert output_tensor.shape == (batches, time_steps, d_model)

    assert not t.allclose(output_tensor[:, 0, :], output_tensor[:, 1, :])

    assert t.allclose(output_tensor[0, 0, :] - input_tensor[0, 0, :],
                      output_tensor[1, 0, :] - input_tensor[1, 0, :])

def test_attention_output_shape(cfg):

    batch_size = 2
    seq_len = 4
    d_model = cfg.d_model

    input = t.ones(batch_size, seq_len, d_model).to(cfg.device)

    attention = Attention(cfg).to(cfg.device)

    assert attention.forward(input).shape == (batch_size, seq_len, d_model)

def test_character_vocabulary_encode_decode(cfg):

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

def test_character_embedding(cfg):

    batch_size = 2
    seq_len = 4
    d_model = cfg.d_model

    input = t.ones(batch_size, seq_len, dtype=t.long).to(cfg.device)

    ce = CharacterEmbedding(cfg).to(cfg.device)

    assert ce.forward(input).shape == (batch_size, seq_len, d_model)

def test_commonvoice_dataset(cfg):
    """Test CommonVoiceDataset with first 100 items"""
    # Initialize dataset with first 100 items
    tsv_path = "/data/jo/commonvoice/cv-corpus-19.0-2024-09-13/en/test.tsv"  # Adjust path as needed
    clips_path = "/data/jo/commonvoice/cv-corpus-19.0-2024-09-13/en/clips_wav"     # Adjust path as needed
    
    dataset = CommonVoiceDataset(cfg, tsv_path, clips_path)
    
    # Test dataset size
    assert len(dataset) > 0, "Dataset should not be empty"
    
    # Test getting an item
    item = dataset[0]
    
    # Test return format
    assert isinstance(item, dict), "Dataset item should be a dictionary"
    assert all(k in item for k in ['audio_features', 'text', 'audio_frames', 'text_length']), "Missing required keys in item"
    
    # Test audio features shape
    audio_features = item['audio_features']
    assert isinstance(audio_features, t.Tensor), "Audio features should be a tensor"
    assert audio_features.dim() == 2, "Audio features should be 2D (time, freq_bins)"
    assert audio_features.shape[1] == cfg.n_freq_bins, f"Should have {cfg.n_freq_bins} frequency bins"
    
    # Test text encoding
    text = item['text']
    assert isinstance(text, list), "Encoded text should be a list"
    assert len(text) == cfg.max_seq_length, f"Text should be padded to {cfg.max_seq_length}"
    
    # Test multiple items
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        #assert item['audio_features'].shape[0] == 3, f"Item {i} should have 3 channels"
        assert len(item['text']) == cfg.max_seq_length, f"Item {i} text should be properly padded"

def test_commonvoice_dataset_processing(cfg):
    """Test audio processing pipeline"""
    tsv_path = "/data/jo/commonvoice/cv-corpus-19.0-2024-09-13/en/test.tsv"  # Adjust path as needed
    clips_path = "/data/jo/commonvoice/cv-corpus-19.0-2024-09-13/en/clips_wav"     # Adjust path as needed
    
    dataset = CommonVoiceDataset(cfg, tsv_path, clips_path)
    
    # Get an item and verify processing
    item = dataset[0]
    features = item['audio_features']
    
    # Test normalization
    assert -10 < features.mean() < 10, "Features should be roughly normalized"
    assert 0 < features.std() < 10, "Features should have reasonable standard deviation"
    
    # Test for NaN values
    assert not t.isnan(features).any(), "Features should not contain NaN values"
    assert not t.isinf(features).any(), "Features should not contain inf values"
    
    # Test delta computations
    # First channel is mel spec, second is first-order delta, third is second-order delta
    #mel_spec = features[0]
    #first_delta = features[1]
    #second_delta = features[2]
    
    #assert not t.allclose(mel_spec, first_delta), "First delta should differ from mel spec"
    #assert not t.allclose(first_delta, second_delta), "Second delta should differ from first delta"

def test_collate_fn(cfg):
    vocab = CharacterVocabulary(cfg)

    batch_samples = [
        {
            'audio_features': t.randn(3, 100, 80),
            'text': vocab.encode('mnemonic games'),
            'audio_frames': 100,
            'text_length': 14,
        },
        {
            'audio_features': t.randn(3, 150, 80),
            'text': vocab.encode('listening machines'),
            'audio_frames': 150,
            'text_length': 18,
        },
        {
            'audio_features': t.randn(3, 200, 80),
            'text': vocab.encode('stars in my pocket'),
            'audio_frames': 200,
            'text_length': 18,
        },
    ]

    collate_fn = create_collate_fn(vocab, cfg)
    batch = collate_fn(batch_samples)

    # test batch structure
    assert 'all_audio_features' in batch
    assert 'all_texts' in batch
    assert 'all_audio_lengths' in batch
    assert 'all_text_lengths' in batch
    assert 'all_audio_masks' in batch
    assert 'all_text_masks' in batch

    # test shapes
    assert len(batch['all_audio_features']) > 0
    assert batch['all_audio_features'][0].shape[0] == len(batch_samples)
    assert batch['all_audio_features'][0].shape[1] == 3

    # test masks

    assert t.all(batch['all_audio_masks'][0][:, 0]) # first frame should be valid for all samples
    assert not t.all(batch['all_audio_masks'][0][:, -1]) # last frame should be padding for some samples

def test_collate_fn_large_batch(cfg):

    vocab = CharacterVocabulary(cfg)

    batch_samples = [
        {
            'audio_features': t.randn(3, 10000, 80),
            'text': vocab.encode('mnemonic games'),
            'audio_frames': 10000,
            'text_length': 14,
        },
        {
            'audio_features': t.randn(3, 3000, 80),
            'text': vocab.encode('listening machines'),
            'audio_frames': 3000,
            'text_length': 18,
        },
        {
            'audio_features': t.randn(3, 8000, 80),
            'text': vocab.encode('stars in my pocket'),
            'audio_frames': 8000,
            'text_length': 18,
        },
    ]

    collate_fn = create_collate_fn(vocab, cfg)
    batch = collate_fn(batch_samples)

    assert len(batch['all_audio_features']) == 2
    

def test_compute_loss(model, cfg):

    def make_prob_dist(high_prob_idx, vocab_size):
        prob_dist = t.ones(vocab_size)

        if high_prob_idx is None:
            return t.fill_(prob_dist, 1/vocab_size)

        prob_dist[high_prob_idx] = 0.8
        other_probs = 0.2 / (vocab_size - 1)
        mask = t.arange(prob_dist.shape[0]) != high_prob_idx
        prob_dist = t.where(mask, t.full_like(prob_dist, other_probs), prob_dist)

        return prob_dist

    prob_dist = make_prob_dist(1, cfg.vocab_size)

    assert t.allclose(prob_dist.sum(), t.tensor(1.0))
    
    stl = SpeechTransformerLearner(cfg)

    probabilities = t.stack([
            t.stack([
                make_prob_dist(1, cfg.vocab_size),
                make_prob_dist(2, cfg.vocab_size),
                make_prob_dist(3, cfg.vocab_size),
                make_prob_dist(4, cfg.vocab_size),
            ]),
            t.stack([
                make_prob_dist(1, cfg.vocab_size),
                make_prob_dist(2, cfg.vocab_size),
                make_prob_dist(None, cfg.vocab_size),
                make_prob_dist(None, cfg.vocab_size),
            ])
        ])

    target_tokens = t.tensor(
        [
            [1, 2, 3, 4],
            [1, 2, 0, 0] # padded sequence
        ]
        )

    padding_mask = t.tensor(
        [
            [True, True, True, True],
            [True, True, False, False]
        ]
        )

    print(f"\nShapes:")
    print(f"probabilities: {probabilities.shape}")
    print(f"target_tokens: {target_tokens.shape}")
    print(f"padding_mask: {padding_mask.shape}")

    print(f"\nValues:")
    print(f"probabilities first position: {probabilities[0, 0, :10]}")  # first 10 values
    print(f"target_tokens: {target_tokens}")
    print(f"padding_mask: {padding_mask}")


    loss = stl.compute_loss(probabilities, target_tokens, padding_mask)

    print("loss: ", loss)
    assert t.all(loss > 0), "Loss should not contain any negative values"
    assert not t.isnan(loss).any(), "Loss should not contain NaN values"
    assert not t.isinf(loss).any(), "Loss should not contain inf values"
                    

"""

Add tests for:

Now that we have the loss working correctly, we could add some more specific test cases to verify:
Loss behavior with perfect predictions
Loss behavior with completely wrong predictions
Effect of padding on loss
Effect of label smoothing (0.8 vs no smoothing)

"""