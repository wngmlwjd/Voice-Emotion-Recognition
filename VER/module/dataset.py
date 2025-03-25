from torch.utils.data import Dataset
import torch
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

class EmotionDataset(Dataset):
    def __init__(self, audio_files, feature_extractor, label_encoder_path=None, labels=None):
        self.audio_files = audio_files
        self.feature_extractor = feature_extractor

        if label_encoder_path and os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        else:
            self.labels = labels
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels)
            if label_encoder_path:
                with open(label_encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoder, f)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        features = self.feature_extractor.extract_features(audio_file).squeeze(0)
        label = self.label_encoder.transform([label])[0]
        return features, label

def collate_fn(batch):
    features = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    features = pad_sequence(features, batch_first=True)
    return features, labels
