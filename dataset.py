import torch
import torchaudio
import pandas as pd



class TrainDataset(torch.utils.data.Dataset):
    """Custom competition dataset."""

    def __init__(self, root='', csv_path='labels_sheila.csv', kw='sheila', transform=None):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            root (string): Directory with all the images.
            kw (string): keyword
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.kw = kw
        self.csv = pd.read_csv(csv_path)
        self.transform = transform


    def __len__(self):
        return self.csv.shape[0]


    def __getitem__(self, idx):
        utt_name = self.root + self.csv.loc[idx, 'name']
        utt = torchaudio.load(utt_name)[0].squeeze()
        word = self.csv.loc[idx, 'word']
        label = self.csv.loc[idx, 'label']

        if self.transform:
            utt = self.transform(utt)

        sample = {'utt': utt, 'word': word, 'label': label}
        return sample
