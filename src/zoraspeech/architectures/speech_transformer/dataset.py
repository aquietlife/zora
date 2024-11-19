import torchaudio
from torch.utils.data import Dataset

class SpeechTransformerDataset(Dataset):
    def __init__(self, file_paths, tokenizer, max_seq_len=1024, transforms=None):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transforms = transforms

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        audio, sr = torchaudio.load(file_path)
        if self.transforms:
            audio = self.transforms(audio)
        return audio
