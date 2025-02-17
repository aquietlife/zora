import einops
import math
import os
import string
from torch.utils.data import Dataset, DataLoader
import torch as t
import pandas as pd
from zoraspeech.architectures.speech_transformer.speech_transformer import Config
import torchaudio

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

        #all_chars = list(string.ascii_lowercase + string.digits + string.punctuation + string.whitespace)
        all_chars = list(string.ascii_lowercase + " ")

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
        
        #print(indices)
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

class CommonVoiceDataset(Dataset):
    def __init__(self, cfg, tsv_path, clips_path):
        self.cfg = cfg
        self.tsv_path = tsv_path
        self.tsv_items =  pd.read_csv(self.tsv_path, sep="\t", low_memory=False)
        self.clips_path = clips_path
        self.cv = CharacterVocabulary(Config())
        self.mel_spec_transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            n_mels=self.cfg.n_freq_bins,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            f_max=self.cfg.f_max
            )
        
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.cfg.n_fft,
            win_length=self.cfg.win_length,
            hop_length=self.cfg.hop_length
        )

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=self.cfg.n_freq_bins,
            sample_rate=self.cfg.sample_rate,
            n_stft = self.cfg.n_fft // 2 + 1,
            f_max=self.cfg.f_max,
            norm='slaney'
        )

        self.compute_deltas_transformation = torchaudio.transforms.ComputeDeltas()

        # replace .mp3 with .wav
        def replace_mp3_with_wav(path):
            return path.replace("mp3", "wav")
        self.tsv_items['path'] = self.tsv_items['path'].apply(replace_mp3_with_wav)

        # clean sentences
        def clean_sentences(s):
            return s.split('\t')[0].strip().lower()
        self.tsv_items['sentence'] = self.tsv_items['sentence'].apply(clean_sentences)

        # filter tsv_items to only contain items with sentences of length < max_seq_length
        self.tsv_items = self.tsv_items[ self.tsv_items['sentence'].str.len() <= self.cfg.max_seq_length ]

        # TODO: group by client_id to perform normalization
        # Remember: The paper used per-speaker normalization because different speakers have different acoustic characteristics (pitch, volume, speaking rate, etc.)

    def __len__(self):
        return len(self.tsv_items)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.clips_path, self.tsv_items.iloc[idx]['path'])
        sentence = self.tsv_items.iloc[idx]['sentence']

        encoded_sentence = self.cv.encode(sentence.lower())

        wav_file, sample_rate = torchaudio.load(audio_path)
        #print(f"Original audio length: {wav_file.shape}, duration: {wav_file.shape[-1]/sample_rate} seconds")

        metadata = torchaudio.info(audio_path)

        #mel_spec= self.mel_spec_transformation(wav_file)

        spec = self.spectrogram(wav_file)
        #print(f"Spectrogram shape: {spec.shape}")  # Should be [1, n_fft//2 + 1, time]

        mel_spec = self.mel_scale(spec)
        #print(f"Mel spec shape: {mel_spec.shape}")
        #print(f"Mel spec shape: {mel_spec.shape}")  # Should be [1, 80, time]

        #first_order_deltas = self.compute_deltas_transformation(mel_spec)

        #second_order_deltas = self.compute_deltas_transformation(first_order_deltas)

        #mel_spec_deltas = t.cat([mel_spec, first_order_deltas, second_order_deltas], dim=1)
        #mel_spec_deltas = t.stack([mel_spec, first_order_deltas, second_order_deltas], dim=0)

        # reorder from (channels, freq_bins, time) to (channels, time, freq_bins)
        #mel_spec_deltas = einops.rearrange(mel_spec_deltas, "1 fb ts -> ts fb")
        #mel_spec_deltas = einops.rearrange(mel_spec_deltas, "f 1 fb t -> t f fb")

        # apply normalization
        #mean = mel_spec_deltas.mean(dim=(0,1), keepdim=True)

        mel_spec = einops.rearrange(mel_spec, "1 fb t -> t fb")

        mean = mel_spec.mean(dim=(0), keepdim=True)
        mel_spec -= mean
        std = mel_spec.std(dim=(0), keepdim=True)
        mel_spec /= std

        #print(metadata.num_frames)
        #print(mel_spec.shape)

        return {
            'audio_features' : mel_spec,
            'text' : encoded_sentence,
            'audio_frames': mel_spec.shape[-1], # number of mel spec frames ( (audio frames / sample rate) * 100 ) - one frame every 10ms 
            'text_length': len(sentence)
        }


def create_collate_fn(vocab: CharacterVocabulary, cfg: Config):
    def collate_fn(batch_samples: list) -> dict:
        """
        Will handle:
            - Dynamic batching to target 20k frames
            - Padding sequences in batch to same length
            - Creating attention masks

        Handles dynamic batching and padding of samples.
    
        Args:
            batch_samples: List of dictionaries containing audio features and text
            
        Returns:
            dict with keys:
                - audio_features: List of tensors [batch_size, channels, time, freq_bins]
                - text: List of tensors [batch_size, seq_length]
                - audio_lengths: List of tensors containing original audio lengths
                - text_lengths: List of tensors containing original text lengths
                - audio_masks: List of boolean tensors for audio padding
                - text_masks: List of boolean tensors for text padding
        """

        # create batches that have up to 20000 frames
        batch_samples.sort(key=lambda x: x['audio_frames'])

        batches = []
        current_batch = []
        total_frames = 0

        for sample in batch_samples:
            #print(sample['audio_frames'])

            if total_frames + sample['audio_frames'] <= cfg.frame_features:
                current_batch.append(sample)
                total_frames += sample['audio_frames']
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [sample] # reset current batch with current sample
                total_frames = sample['audio_frames'] # reset total_frames with current sample's current frames

        if current_batch:
            batches.append(current_batch)

        all_audio_features = []
        all_texts = []
        all_audio_lengths = []
        all_text_lengths = []

        all_audio_masks = []
        all_text_masks = []
        
        # pad each sample
        for batch in batches:
            longest_audio_sequence = max(sample['audio_frames'] for sample in batch)
            padded_features = []
            audio_lengths = []

            longest_text_sequence = max(len(sample['text']) for sample in batch)
            padded_texts = []
            text_lengths = []

            # pad audio sequence
            for sample in batch:
                # audio padding
                features = sample['audio_features']
                audio_padding_length = longest_audio_sequence - features.shape[1]
                padded = t.nn.functional.pad(features, (0, 0, 0, audio_padding_length))
                padded_features.append(padded)
                audio_lengths.append(sample['audio_frames'])

                # text padding
                text = sample['text']
                text_len = len(text)
                text_padding_length = longest_text_sequence - text_len
                padded = text + [vocab.char_to_idx['PAD']] * text_padding_length
                padded_texts.append(padded)
                text_lengths.append(text_len)


            batch_features = t.stack(padded_features)
            all_audio_features.append(batch_features)
            all_audio_lengths.append(t.tensor(audio_lengths))

            batch_texts = t.tensor(padded_texts)
            all_texts.append(batch_texts)
            all_text_lengths.append(t.tensor(text_lengths))

            # add attention masks so that transformer doesn't learn from padding tokens
            audio_mask = t.arange(longest_audio_sequence)[None, :] < t.tensor(audio_lengths)[:, None].clone().detach()
            text_mask = t.arange(longest_text_sequence)[None, :] < t.tensor(text_lengths)[:, None].clone().detach()
            all_audio_masks.append(audio_mask)
            all_text_masks.append(text_mask)
        

        
        return {
            'all_audio_features': all_audio_features,
            'all_texts': all_texts,
            'all_audio_lengths': all_audio_lengths,
            'all_text_lengths': all_text_lengths,
            'all_audio_masks': all_audio_masks,
            'all_text_masks': all_text_masks
        }
    return collate_fn
        





