from zoraspeech.architectures.speech_transformer.speech_transformer import SpeechTransformer, Config
import einops
import torch as t
from jaxtyping import Float, Int, Bool

class SpeechTransformerLearner:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = SpeechTransformer(self.cfg)
        # loss function ??
        self.optimizer = t.optim.AdamW(
            self.model.parameters(),
            lr=1e-2, # how do we implement this variable learning rate? can we just try a fixed value for now?
            betas=(self.cfg.op_beta_1, self.cfg.op_beta_2),
            eps=self.cfg.op_eps
        )
        self.steps = 0
        self.use_wandb = True

    def compute_loss(
              self,
              probabilities: Float[t.Tensor, "batch seq_length d_vocab"], #type: ignore
              tokens: Int[t.Tensor, "batch seq_length"], #type: ignore
              padding_mask: Bool[t.Tensor, "batch seq_length"] #type: ignore
              ) -> Float[t.Tensor, "batch seq_length-1"]: #type: ignore
        """
        Usage: loss = -calculate_log_probs(logits, inputs).mean()
        GPT-2 Implementation for reference
        log_probs = logits.log_softmax(dim=-1)
        log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

        Input: logits, encoded text tokens
        Output: negative log probabilities of the first seq_length - 1 predictions (so we can compare them with the actual next tokens)
        
        Compute cross entropy loss with:
        - Label smoothing (0.8 for correct token, 0.2/(V-1) for others)
        - Padding mask to ignore pad tokens
        - Shifted prediction (predict next token)
        """
        # handle teacher forcing alignment
        probabilities = probabilities[:, :-1, :]
        tokens = tokens[:, 1:]
        padding_mask = padding_mask[:, 1:]

        print("After alignment:")
        print(f"probabilities: {probabilities.shape}")
        print(f"tokens: {tokens.shape}")
        print(f"padding_mask: {padding_mask.shape}")

        # apply label smoothed target distribution
        smoothed_target_distribution = t.zeros(probabilities.shape).to(probabilities.device)
        smoothed_target_distribution.fill_(0.2 / (self.cfg.vocab_size - 1))
        indices = einops.rearrange(tokens, "b s -> b s 1")
        smoothed_target_distribution.scatter_(-1, indices, 0.8)

        assert (smoothed_target_distribution.sum(dim=-1) - 1.0).abs().max() < 1e-6

        # compute cross entry with smoothed targets
        log_probs = t.log(probabilities + 1e-10) # small eps to prevent log(0)
        log_probs = t.mul(log_probs, smoothed_target_distribution)
        sum = log_probs.sum(dim=-1)
        masked_sum = sum.masked_fill_(~padding_mask, 0) # invert mask so we zero out padded tokens (which come in as False)
        loss = masked_sum.mean()        

        return -loss


    def learning_step():
        # learning step happens here
        pass

    def learn(self):
        # training loop happens here
        pass

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass


    def get_learning_rate(self):
        pass