from dataclasses import dataclass
from jaxtyping import Float
import torch as t
import torch.nn as nn
import einops
import string

# config class to hold hyperparamters
@dataclass
class Config:
    debug: bool = True
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    n_freq_bins = 80 # number of frequency bins from the spectrograms, defined in paper
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

    # character embeddings
    vocab_size: int = 78
    pad_idx: int = 0
    sos_idx: int = 1 # start of sequence
    eos_idx: int = 2 # end of sequence
    unk_idx: int = 3 # unknown token
    include_special_chars: bool =  False
    include_sos_eos_tokens: bool = True
    unknown_char: str = "unknown_char"
    max_seq_length: int = 100

class SpeechTransformer(nn.Module):
    """Speech Transformer model that converts speech spectrograms to text.
    
    The model follows the architecture from "Speech-Transformer: A No-Recurrence 
    Sequence-to-Sequence Model for Speech Recognition" paper.
    
    Architecture Overview:
    1. Two Conv2d layers with stride 2 reduce time and frequency dimensions by 4x
    2. Linear projection to d_model dimension
    3. Positional encoding
    4. Transformer encoder blocks
    5. Layer norm
    
    Currently encoder-only!

    Input shape: [batch_size, time_steps, freq_bins]
    Output shape: [batch_size, reduced_time_steps, d_model]
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config()

        # encoder components
        self.repeat = Repeat(self.cfg)
        self.conv2d_block_one = Conv2DBlock(self.cfg, self.cfg.n_channels, self.cfg.n_out_channels, self.cfg.conv2d_kernel_size, self.cfg.conv2d_stride, self.cfg.conv2d_padding)
        self.conv2d_block_two = Conv2DBlock(self.cfg, self.cfg.n_out_channels, self.cfg.n_out_channels, self.cfg.conv2d_kernel_size, self.cfg.conv2d_stride, self.cfg.conv2d_padding)
        self.reshape = Reshape(self.cfg, "b c ts fb -> b ts (c fb)")
        self.linear = Linear(self.cfg)
        self.encoder_positional_encoder = PositionalEncoder(self.cfg)
        self.encoder_blocks = nn.Sequential(
            *[EncoderBlock() for _ in range(self.cfg.n_encoder_layers)]
        )
        self.layer_norm = LayerNorm(self.cfg)

        # encoder as an nn.Sequential
        self.encoder = nn.Sequential(
            self.repeat,
            self.conv2d_block_one,
            self.conv2d_block_two,
            self.reshape,
            self.linear,
            self.encoder_positional_encoder,
            self.encoder_blocks,
            self.layer_norm
        )


        # decoder as a custom module
        self.decoder = Decoder(self.cfg)

    def forward(self, speech_input, text_input):
        encoder_output = self.encoder(speech_input)
        decoder_output = self.decoder(text_input, key_input=encoder_output, value_input=encoder_output)
        return decoder_output

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
            value_input: Optional[Float[t.Tensor, "batch, posn d_model"]] = None # type: ignore
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




class CharacterVocabulary:
    """Handles character-level tokenization for the Speech Transformer.
    
    Maps between characters and integer indices, handles special tokens (PAD, SOS, EOS, UNK),
    and manages sequence padding/truncation.
    
    Attributes:
        cfg: Configuration object containing vocabulary parameters
        char_to_idx: Dictionary mapping characters to integer indices
        idx_to_char: Dictionary mapping integer indices to characters
        special_tokens: List of special token indices [PAD, SOS, EOS, UNK]
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        if self.cfg.max_seq_length <= 0 or self.cfg.max_seq_length > 1000:
            raise ValueError("max_seq_length is either too small or large!")

        self.special_tokens = [
            self.cfg.pad_idx, 
            self.cfg.sos_idx, 
            self.cfg.eos_idx, 
            self.cfg.unk_idx
            ]

        self.char_to_idx = dict() # maps chars to integer index
        self.idx_to_char = dict() # maps interger index to chars

        # initialize char to idx dict
        # with special tokens first
        self.char_to_idx['PAD'] = self.cfg.pad_idx
        self.char_to_idx['SOS'] = self.cfg.sos_idx
        self.char_to_idx['EOS'] = self.cfg.eos_idx
        self.char_to_idx['UNK'] = self.cfg.unk_idx

        all_chars = list(string.ascii_lowercase + string.digits + string.punctuation + string.whitespace)

        for i, c in enumerate(all_chars):
            self.char_to_idx[c] = i + 4 # offset by 4 to account for special characters we manually added first

        # initialize idx to char dict (essentially a reverse mapping of char_to_idx)
        self.idx_to_char = { value: key for key, value in self.char_to_idx.items() }
        
        assert len(list(self.char_to_idx.keys())) == self.cfg.vocab_size

    def encode(self, s: str) -> list:
        """Converts a string to a list of token indices.
        
        Handles unknown characters, adds SOS/EOS tokens if configured,
        and pads/truncates to max_seq_length.
        
        Args:
            s: Input string to encode
            
        Returns:
            List of indices with optional SOS/EOS tokens and padding
            
        Raises:
            ValueError: If input is None or empty string
        """
        
        if not s:
            raise ValueError("encode expected a string but got None or an empty string") 

        indices = []

        for c in s:
            # find index in char_to_idx dict
            if c not in self.char_to_idx:
                idx = self.cfg.unk_idx
            else:
                idx = self.char_to_idx[c]
            indices.append(idx)
        
        # add sos and eos tokens
        if self.cfg.include_sos_eos_tokens:
            indices = [self.cfg.sos_idx] + indices + [self.cfg.eos_idx]

        # add padding if necessary

        indices = self.pad_sequence(indices)

        return indices

    def decode(self, indices: list) -> str:
        """Converts a list of token indices back to a string.
        
        Handles special tokens based on configuration, converts unknown
        tokens to unknown_char.
        
        Args:
            indices: List of token indices to decode
            
        Returns:
            Decoded string with special tokens optionally removed
            
        Raises:
            ValueError: If indices is None or empty list
        """
        
        if not indices:
            raise ValueError("decode expect a list but got None or an empty list")
        
        # convert indices to string
        chars = []

        for i in indices:
            if not self.cfg.include_special_chars:
                if i in self.special_tokens:
                    continue
            char = self.idx_to_char[i]
            if i == self.cfg.unk_idx:
                char = self.cfg.unknown_char
            chars.append(char)

        return "".join(chars)

    def pad_sequence(self, indices: list) -> list:
        """Pads or truncates sequence to configured max_seq_length.
        
        Args:
            indices: List of token indices
            
        Returns:
            Padded/truncated list of length max_seq_length
        """
        
        if len(indices) < self.cfg.max_seq_length:

            # figure out how much we are off by
            diff = self.cfg.max_seq_length - len(indices)

            # pad to the right
            indices = indices + [self.cfg.pad_idx for _ in range(diff)]

        elif len(indices) > self.cfg.max_seq_length:
            indices = indices[:self.cfg.max_seq_length]
        return indices

    def add_sos_eos(self, indices: list) -> list:
        """Adds start and end of sequence tokens to list of indices.
        
        Args:
            indices: List of token indices
            
        Returns:
            List with SOS token prepended and EOS token appended
        """
        
        return [self.cfg.sos_idx] + indices + [self.cfg.eos_idx]

    def is_special_token(self, idx: int) -> bool:
        """Checks if an index corresponds to a special token.
        
        Args:
            idx: Token index to check
            
        Returns:
            True if idx is a special token (PAD, SOS, EOS, UNK)
        """
        
        return True if idx in self.special_tokens else False

    @property
    def vocab_size(self) -> int:
        """Total size of vocabulary including special tokens."""
        
        return len(self.char_to_idx)
    
    def get_vocab(self) -> list:
        """Returns list of all characters in vocabulary."""
        
        return list(self.char_to_idx.keys())

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
            value_input: Optional[Float[t.Tensor, "batch, posn d_model"]] = None # type: ignore
            ) -> Float[t.Tensor, "batch posn d_model"]: # type: ignore

         x = self.add_to_residual_stream(x, self.layer_norm_one, self.attention_masked)
         x = self.add_to_residual_stream(x, self.layer_norm_two, lambda norm_x: self.attention_unmasked(norm_x, key_input, value_input)) 
         x = self.add_to_residual_stream(x, self.layer_norm_three, self.feed_forward_network) # this layer needs to use encoder outputs as its inputs for keys and values, and use queries from previous sub-block outputs
        
         return x