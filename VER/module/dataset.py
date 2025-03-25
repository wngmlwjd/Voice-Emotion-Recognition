from torch.utils.data import Dataset
import torch
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

class EmotionDataset(Dataset):
    def __init__(self, audio_files, labels, feature_extractor, label_encoder_path):
        self.audio_files = audio_files
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        
        # LabelEncoder 초기화
        if label_encoder_path and os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"LabelEncoder loaded from {label_encoder_path}")
        else:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels)
            if label_encoder_path:
                with open(label_encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                print(f"LabelEncoder saved to {label_encoder_path}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        # HuBERT로 음성 특성 추출
        features = self.feature_extractor.extract_features(audio_file)
        features = features.squeeze(0)
        
        # 라벨 인코딩
        label = self.label_encoder.transform([label])[0]
        
        return features, label

def collate_fn(batch):
    features = [item[0] for item in batch]  # 각 샘플의 feature 추출
    labels = torch.tensor([item[1] for item in batch])  # 각 샘플의 label 추출
    
    # 시퀀스 패딩 적용 (seq_len을 동일하게 맞춤)
    features = pad_sequence(features, batch_first=True)  # (batch_size, max_seq_len, feature_dim)
    
    return features, labels