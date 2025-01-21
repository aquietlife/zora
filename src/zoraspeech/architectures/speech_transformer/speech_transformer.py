from dataclasses import dataclass
from jaxtyping import Float
import torch as t
import torch.nn as nn
import einops

# config class to hold hyperparamters
# TODO annotate what each of these hyperparameters means
@dataclass
class Config:
    debug: bool = True
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    n_freq_bins = 80
    n_channels = 1
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

class SpeechTransformer(nn.Module):
    """Speech Transformer model that converts speech spectrograms to text.
    
    The model follows the architecture from "Speech-Transformer: A No-Recurrence 
    Sequence-to-Sequence Model for Speech Recognition" paper.
    
    Architecture Overview:
    1. Two Conv2d layers with stride 2 reduce time and frequency dimensions by 4x
    2. Linear projection to d_model dimension
    3. Positional encoding
    4. Transformer encoder blocks
    
    Input shape: [batch_size, time_steps, freq_bins]
    Output shape: [batch_size, reduced_time_steps, d_model]
    """

    def __init__(self):
        super().__init__()
        self.cfg = Config()

        self.repeat = Repeat()
        self.conv2d_block_one = Conv2DBlock(self.cfg, self.cfg.n_channels, self.cfg.n_out_channels, self.cfg.conv2d_kernel_size, self.cfg.conv2d_stride, self.cfg.conv2d_padding)
        self.conv2d_block_two = Conv2DBlock(self.cfg, self.cfg.n_out_channels, self.cfg.n_out_channels, self.cfg.conv2d_kernel_size, self.cfg.conv2d_stride, self.cfg.conv2d_padding)
        self.reshape = Reshape(self.cfg, "b c ts fb -> b ts (c fb)")
        self.linear = Linear(self.cfg)
        self.positional_encoder = PositionalEncoder(self.cfg)
        self.encoder_blocks = nn.Sequential(
            *[EncoderBlock() for _ in range(self.cfg.n_encoder_layers)]
        )
        self.layer_norm = LayerNorm(self.cfg)

        # encoder
        self.encoder = nn.Sequential(
            self.repeat,
            self.conv2d_block_one,
            self.conv2d_block_two,
            self.reshape,
            self.linear,
            self.positional_encoder,
            self.encoder_blocks,
            self.layer_norm
        )

        # decoder
        # TBD NEXT

    def forward(self, x: Float[t.Tensor, "batch time_steps freq_bins"]) -> Float[t.Tensor, "batch reduced_time d_model"]: # type: ignore
        """Transform input spectrogram through the Speech Transformer.
        
        The forward pass consists of:
        1. Reshape input to [batch, channels=1, time_steps, freq_bins]
        2. Two Conv2d + ReLU layers that each reduce dimensions by 2x
           - After Conv1: [batch, channels, time_steps/2, freq_bins/2] 
           - After Conv2: [batch, channels, time_steps/4, freq_bins/4]
        3. Linear projection to d_model dimension
        4. Add positional encoding
        5. Process through transformer encoder blocks

        Args:
            x: Input spectrogram of shape [batch, time_steps, freq_bins]
               where freq_bins=80 (from the paper's filterbank features)
        
        Returns:
            Encoded sequence of shape [batch, reduced_time, d_model]
            where reduced_time = time_steps/4 due to the strided convolutions
        """
        return self.encoder(x)

class Conv2DBlock(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.cfg = cfg
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        assert x.shape[1] == self.cfg.n_out_channels

        return x
    
class Linear(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(cfg.n_freq_bins//4 * cfg.n_out_channels, cfg.d_model)

    def forward(self, x):
        x = self.linear(x)
        assert x.shape[2] == self.cfg.d_model
        return x

class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x):
        x = self.layer_norm(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: Float[t.Tensor, "batch reduced_time_steps d_model"]) -> Float[t.Tensor, "batch posn d_model"]: #type: ignore

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

        return x + pos_encoding

class Reshape(nn.Module):
    def __init__(self, cfg, pattern):
        super().__init__()
        self.pattern = pattern
        self.cfg = cfg

    def forward(self, x):
        x = einops.rearrange(x, self.pattern)
        assert x.shape[2] == self.cfg.n_out_channels * (self.cfg.n_freq_bins // 4)
        return x

class Repeat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = einops.repeat(x, "b ts fb -> b c ts fb", c=1)
        assert x.shape[1] == 1
        #assert x.shape[2] == input_time_steps
        #assert x.shape[3] == self.cfg.n_freq_bins
        return x

class FFN(nn.Module):
    def __init__(self, dff, d_model):
        super().__init__()
        self.dff = dff
        self.d_model = d_model

        self.linear_one = nn.Linear(self.d_model, self.dff)
        self.relu = nn.ReLU()
        self.linear_two = nn.Linear(self.dff, self.d_model)

    def forward(self, x: Float[t.Tensor, "batch posn d_model"]) -> Float[t.Tensor, "batch posn d_model"]: # type: ignore
        x = self.linear_one(x)
        x = self.relu(x)
        x = self.linear_two(x)

        assert x.shape[-1] == self.d_model

        return x

class TransformerBlock(nn.Module):
    @staticmethod
    def add_to_residual_stream(x, layer_norm, sub_block):
        normalized = layer_norm(x)
        transformed = sub_block(normalized)
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
        self.attention = Attention(self.cfg, apply_mask=False)
        self.feed_forward_network = FFN(self.cfg.dff, self.cfg.d_model)

    def forward(self, x: Float[t.Tensor, "batch posn d_model"]) -> Float[t.Tensor, "batch posn d_model"]: # type: ignore
        x = self.add_to_residual_stream(x, self.layer_norm_one, self.attention)
        x = self.add_to_residual_stream(x, self.layer_norm_two, self.feed_forward_network)
        return x


class DecoderBlock(TransformerBlock):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

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
            normalized_resid_pre: Float[t.Tensor, "batch posn d_model"] # type: ignore
            ) -> Float[t.Tensor, "batch posn d_model"]: # type: ignore
        # linear map


        Q = einops.einsum(normalized_resid_pre,
                          self.W_Q,
                          "b s e, n e h -> b s n h") + self.b_Q
        
        K = einops.einsum(normalized_resid_pre,
                          self.W_K,
                          "b s e, n e h -> b s n h") + self.b_K
        
        V = einops.einsum(normalized_resid_pre,
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


