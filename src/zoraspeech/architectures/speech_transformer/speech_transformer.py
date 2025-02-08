from dataclasses import dataclass
from jaxtyping import Float
from typing import Optional
import torch as t
import torch.nn as nn
import einops
import string

# config class to hold hyperparamters
@dataclass
class Config:

    debug: bool = True
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    sample_rate = 16000
    n_fft = 512
    n_freq_bins = 80 # number of frequency bins from the spectrograms, defined in paper
    hop_length = (10 * sample_rate) // 1000
    win_length = (25 * sample_rate) // 1000
    f_max = sample_rate // 2
    frame_features = 20000 # testing at 2000, but should be 20000
    n_channels = 1 # number of extra channels for when we pass through conv2d blocks
    n_out_channels = 64 # from the paper
    conv2d_kernel_size = 3
    conv2d_stride = 2
    conv2d_padding = 1
    d_model: int = 256 # dimension of the feature vector that represents each posiiton in the sequence
    n_encoder_layers = 6
    n_decoder_layers = 6 
    layer_norm_eps: float = 1e-5
    init_range: float = 0.02
    d_head: int = 64
    n_heads: int = 4
    dff = 1024

    assert d_model == d_head * n_heads

    # character embeddings
    vocab_size: int = 31
    pad_idx: int = 0
    sos_idx: int = 1 # start of sequence
    eos_idx: int = 2 # end of sequence
    unk_idx: int = 3 # unknown token
    include_special_chars: bool =  False
    include_sos_eos_tokens: bool = True
    unknown_char: str = "unknown_char"
    max_seq_length: int = 100

    # training
    batch_size: int = 1 # doesn't need to be a number like 32 since collate_fn() handles batching for us
    num_workers: int = 4
    shuffle: bool = True
    prefetch_factor: int = 2
    training_steps: int = 20 # turn up to 100000 when doing full training
    neighborhood_smoothing: float = 0.8
    residual_dropout: float = 0.1
    attention_dropout: float = 0.1
    checkpoint_frequency: int = 10
    grad_clip_value: float = 1.0
    validation_frequency: int = 5 # turn up to 100 when doing full training

    # optimizer
    op_beta_1: float = 0.9
    op_beta_2: float = 0.98
    op_eps:float = 10e-9

    # learning rate
    k_start: int = 10
    k_end: int = 1
    k_fixed: int = (k_start + k_end) // 2
    warmup_n: int = 25000

    # beam search
    beam_size: int  = 10
    length_penalty_alpha: int = 1.0

class SpeechTransformer(nn.Module):
    """Speech Transformer model that converts speech spectrograms to text.
    
    The model follows the architecture from "Speech-Transformer: A No-Recurrence 
    Sequence-to-Sequence Model for Speech Recognition" paper.    
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)

    def forward(self, speech_input, text_input):
        """
        Input: speech_input as (batch time_steps freq_bins) - spectrograms and text_input as 
        Output shape: probabilities as (batches, max_seq_length, vocab_size)
        """
        encoder_output = self.encoder(speech_input)
        decoder_output = self.decoder(text_input, key_input=encoder_output, value_input=encoder_output)
        return decoder_output

### ENCODER ###

class Encoder(nn.Module):
    """Encoder

    Input: [batch time_steps freq_bins]
    Output: [batch seq_length d_model]
    
    This class processes a batch of spectrograms by:

    - Expanding the input tensor by one dimension for channel
    - Passing our tensor through a Conv2d block (with ReLU)
    - Passing our tensor through another Conv2d block (with ReLU)
    - Reshaping out tensor so it has three dimensions instead of four
    - Passing through a linear layer so we get an output shape of d_model for the last dimension
    - Passing our tensor through a posiitonal encoder
    - Passing out query_input through n encoder blocks
    - Passing our tensor through a layer norm
    - Outputting our encoded output to be used by the decoder (for cross attention)
    
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # encoder components
        self.repeat = Repeat(self.cfg)
        self.conv2d_block_one = Conv2DBlock(self.cfg, self.cfg.n_channels, self.cfg.n_out_channels, self.cfg.conv2d_kernel_size, self.cfg.conv2d_stride, self.cfg.conv2d_padding)
        self.conv2d_block_two = Conv2DBlock(self.cfg, self.cfg.n_out_channels, self.cfg.n_out_channels, self.cfg.conv2d_kernel_size, self.cfg.conv2d_stride, self.cfg.conv2d_padding)
        self.reshape = Reshape(self.cfg, "b c ts fb -> b ts (c fb)")
        self.linear = Linear(self.cfg)
        self.positional_encoder = PositionalEncoder(self.cfg)
        self.encoder_blocks = nn.Sequential(
            *[EncoderBlock() for _ in range(self.cfg.n_encoder_layers)]
        )
        self.layer_norm = LayerNorm(self.cfg)

        # encoder as an nn.Sequential
        self.sequential = nn.Sequential(
            self.repeat,
            self.conv2d_block_one,
            self.conv2d_block_two,
            self.reshape,
            self.linear,
            self.positional_encoder,
            self.encoder_blocks,
            self.layer_norm
        )

    def forward(self, x: Float[t.Tensor, "batch time_steps freq_bins"]) -> Float[t.Tensor, "batch seq_length d_model"] :  # type: ignore
        # take in our input spectrograms, encode, and generate encoded outputs to be used by the decoder for cross-attention
        print("input shape: ", x.shape)
        assert x.ndim == 3, f"Expected 3 dimensions, got {x.ndim}"
        assert x.shape[2] == self.cfg.n_freq_bins
        return self.sequential(x)

class Conv2DBlock(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.cfg = cfg
        self.out_channels = out_channels
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        assert x.shape[1] == self.out_channels

        return x
    
class Linear(nn.Module):
    """Wraps PyTorch's Linear so we can add assertions
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(cfg.n_freq_bins//4 * cfg.n_out_channels, cfg.d_model)

    def forward(self, x):
        x = self.linear(x)
        assert x.shape[2] == self.cfg.d_model
        return x

class LayerNorm(nn.Module):
    """Wraps PyTorch's LayerNorm so we can add assertions
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x):
        x = self.layer_norm(x)
        assert x.shape[2] == self.cfg.d_model
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: Float[t.Tensor, "batch seq_length d_model"]) -> Float[t.Tensor, "batch seq_length d_model"]: #type: ignore

        # get sequence length from input x
        seq_length = x.shape[1]
        position_indices = t.arange(0, seq_length, device=x.device )
        dimension_indices = t.arange(0, self.cfg.d_model, device=x.device)

        exp = (dimension_indices * 2) / self.cfg.d_model

        angle_rates = t.pow(10000, exp)

        angles = t.outer(position_indices, 1 / angle_rates)

        # create a positional encoding tensor
        pos_encoding = t.zeros_like(angles) # [seq_length, d_model]

        # apply sin to even indices and cos to odd indices
        pos_encoding[:, 0::2] = t.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = t.cos(angles[:, 1::2])

        pos_encoding = einops.repeat(pos_encoding, "seq_length d_model -> batch seq_length d_model", batch=x.shape[0])

        assert x.ndim == 3
        assert x.shape[-1] == self.cfg.d_model

        return x + pos_encoding # this is where our embeddings get added to our positional encoding

class Reshape(nn.Module):
    """Reshape can take in any einops rearrange pattern and return a rearranged output tensor

    e.g. b c ts fb -> b ts (c fb) - This takes in a tensor of (b c ts fb) and reduces it by one dimension by multiplying two together

    """
    def __init__(self, cfg, pattern):
        super().__init__()
        self.pattern = pattern
        self.cfg = cfg

    def forward(self, x):
        x = einops.rearrange(x, self.pattern)
        assert x.shape[2] == self.cfg.n_out_channels * (self.cfg.n_freq_bins // 4)
        return x

class Repeat(nn.Module):
    """Repeat is used to add a new dimension to a tensor, similar to t.unsqueeze()
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        x = einops.repeat(x, "b ts fb -> b c ts fb", c=self.cfg.n_channels)
        assert x.shape[1] == 1
        assert x.shape[3] == self.cfg.n_freq_bins
        return x

class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.linear_one = nn.Linear(self.cfg.d_model, self.cfg.dff)
        self.relu = nn.ReLU()
        self.linear_two = nn.Linear(self.cfg.dff, self.cfg.d_model)

    def forward(self, x: Float[t.Tensor, "batch posn d_model"]) -> Float[t.Tensor, "batch posn d_model"]: # type: ignore
        x = self.linear_one(x)
        x = self.relu(x)
        x = self.linear_two(x)

        assert x.shape[-1] == self.cfg.d_model

        return x

class TransformerBlock(nn.Module):
    """This base class allows both EncoderBlock and DecoderBlock to use add_to_residual_stream method

    This method performs the functionality of the transformer block (where we have attention and FFN),
    making sure to add the result to a residual stream 

    """
    @staticmethod
    def add_to_residual_stream(
        x: Float[t.Tensor, "batch posn d_model"], # type: ignore 
        layer_norm, 
        sub_block_fn) -> Float[t.Tensor, "batch posn d_model"]: # type: ignore
        
        normalized = layer_norm(x)
        transformed = sub_block_fn(normalized)
        output = x + transformed
        return output

class EncoderBlock(TransformerBlock):
    """Encoder block for the Speech Transformer.
    
    Each encoder block contains:
    1. Layer normalization + Multi-head attention (without masking)
    2. Layer normalization + Feed-forward network
    3. Residual connections around each sub-block
    
    Input/Output shape: [batch, posn, d_model]
    """
    def __init__(self):
        super().__init__()
        self.cfg = Config()
        self.layer_norm_one = nn.LayerNorm(self.cfg.d_model)
        self.layer_norm_two = nn.LayerNorm(self.cfg.d_model)
        self.attention_unmasked = Attention(self.cfg, apply_mask=False)
        self.feed_forward_network = FFN(self.cfg)

    def forward(self, x: Float[t.Tensor, "batch posn d_model"]) -> Float[t.Tensor, "batch posn d_model"]: # type: ignore
        x = self.add_to_residual_stream(x, self.layer_norm_one, self.attention_unmasked)
        x = self.add_to_residual_stream(x, self.layer_norm_two, self.feed_forward_network)
        return x

class Attention(nn.Module):

    def __init__(self, cfg: Config, apply_mask: bool = True):
        super().__init__()
        self.cfg = cfg

        self.apply_mask = apply_mask

        # weights
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))

        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
    
        # biases
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))

        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))

        # initialize weights
        nn.init.normal_(self.W_Q, std=cfg.init_range)
        nn.init.normal_(self.W_K, std=cfg.init_range)
        nn.init.normal_(self.W_V, std=cfg.init_range)
        nn.init.normal_(self.W_O, std=cfg.init_range)

        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=self.cfg.device))

    
    def forward(
            self, 
            query_input: Float[t.Tensor, "batch posn d_model"], # type: ignore
            key_input: Optional[Float[t.Tensor, "batch posn d_model"]] = None, # type: ignore
            value_input: Optional[Float[t.Tensor, "batch posn d_model"]] = None # type: ignore
            ) -> Float[t.Tensor, "batch posn d_model"]: # type: ignore
        # linear map


        Q = einops.einsum(query_input,
                          self.W_Q,
                          "b s e, n e h -> b s n h") + self.b_Q
        
        K = einops.einsum(query_input if key_input is None else key_input,
                          self.W_K,
                          "b s e, n e h -> b s n h") + self.b_K
        
        V = einops.einsum(query_input if value_input is None else value_input,
                          self.W_V,
                          "b s e, n e h -> b s n h") + self.b_V
        
        attn_scores = einops.einsum(Q, 
                                    K, 
                                    "batch seq_q head_index d_head, batch seq_k head_index d_head -> batch head_index seq_q seq_k"
                                    )
        
        attn_scores = attn_scores / (self.cfg.d_head ** 0.5) # scale

        if self.apply_mask:
            attn_scores = self.apply_causal_mask(attn_scores)

        A = t.softmax(attn_scores, dim=-1) # attention is all we need!

        z = einops.einsum(A, V, "b n sq sk, b sk n h -> b sq n h")

        result = einops.einsum(z, self.W_O, "b sq n h, n h e -> b sq e")

        return result + self.b_O
    
    def apply_causal_mask(self, attn_scores: Float[t.Tensor, "batch n_heads query_pos key_pos"] # type: ignore
                          ) -> Float[t.Tensor, "batch n_heads query_pos key_pos"]: # type: ignore
        mask = t.triu(t.ones_like(attn_scores), diagonal = 1).to(self.cfg.device)
        return attn_scores.masked_fill_(mask != 0, self.IGNORE)

### DECODER ###

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.character_embedding = CharacterEmbedding(self.cfg)
        self.decoder_positional_encoder = PositionalEncoder(self.cfg)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(self.cfg) for _ in range(self.cfg.n_decoder_layers)]
        )
        self.layer_norm = LayerNorm(self.cfg)
        self.linear = nn.Linear(self.cfg.d_model, self.cfg.vocab_size)
        self.soft_max = nn.Softmax(dim=-1)  # Apply softmax along the vocabulary dimension


    def forward(self, x,
            key_input: Optional[Float[t.Tensor, "batch posn d_model"]] = None, # type: ignore
            value_input: Optional[Float[t.Tensor, "batch posn d_model"]] = None # type: ignore
    ):
        # take in our input tokens, encode, and generate character embeddings
        embeddings = self.character_embedding(x)

        encoded_sequence = self.decoder_positional_encoder(embeddings)

        assert encoded_sequence.ndim == 3
        assert encoded_sequence.shape[1] <= self.cfg.max_seq_length
        assert encoded_sequence.shape[2] == self.cfg.d_model
        
        # pass positional encoding into decoder blocks
        x = encoded_sequence
        for block in self.decoder_blocks:
            x = block(x, key_input, value_input)

        x = self.layer_norm(x)
        
        x = self.linear(x)
        
        probabilities = self.soft_max(x)

        return probabilities

class CharacterEmbedding(nn.Module):
    """Character embedding layer for the Speech Transformer decoder.
    
    Converts token indices from CharacterVocabulary into dense vectors of dimension d_model.
    This is the first component in the decoder sequence, followed by positional encoding.
    Padding tokens are passed through and handled later by attention masks.
    
    Input shape: [batch_size, seq_length] - Tensor of token indices from CharacterVocabulary
    Output shape: [batch_size, seq_length, d_model] - Dense embedding vectors
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.character_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        nn.init.normal_(self.character_embedding.weight, std=cfg.init_range)
    
    def forward(self, x: Float[t.Tensor, "batch seq_length"]) -> Float[t.Tensor, "batch seq_length d_model"]: # type: ignore
        assert x.shape[1] <= self.cfg.max_seq_length
        assert t.all( x < self.cfg.vocab_size) 
        assert len(x.shape) == 2

        x = self.character_embedding(x)
        
        assert len(x.shape) == 3
        assert x.shape[2] == self.cfg.d_model

        return x

class DecoderBlock(TransformerBlock):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layer_norm_one = nn.LayerNorm(self.cfg.d_model)
        self.attention_masked = Attention(self.cfg, apply_mask=True)
        self.layer_norm_two = nn.LayerNorm(self.cfg.d_model)
        self.attention_unmasked = Attention(self.cfg, apply_mask=False)
        self.layer_norm_three = nn.LayerNorm(self.cfg.d_model)
        self.feed_forward_network = FFN(self.cfg)

    def forward(
            self, 
            x: Float[t.Tensor, "batch posn d_model"], # type: ignore
            key_input: Optional[Float[t.Tensor, "batch posn d_model"]] = None, # type: ignore
            value_input: Optional[Float[t.Tensor, "batch posn d_model"]] = None # type: ignore
            ) -> Float[t.Tensor, "batch posn d_model"]: # type: ignore

        x = self.add_to_residual_stream(x, self.layer_norm_one, self.attention_masked)
        x = self.add_to_residual_stream(x, self.layer_norm_two, lambda norm_x: self.attention_unmasked(norm_x, key_input, value_input)) # this layer needs to use encoder outputs as its inputs for keys and values, and use queries from previous sub-block outputs
        x = self.add_to_residual_stream(x, self.layer_norm_three, self.feed_forward_network)
        
        return x