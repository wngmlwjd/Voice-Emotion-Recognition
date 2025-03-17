from module.feature_extraction import HuBERTFeatureExtractor
from module.dataset import EmotionDataset, collate_fn
from module.model import EmotionTransformer
from module.train import train_model
from module.utils import get_file_names, get_folder_names
from module.config import *

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import torchaudio

def main():
    # torchaudio.set_audio_backend("sox_io")
    # print(torchaudio.get_audio_backend())
    
    feature_extractor = HuBERTFeatureExtractor()

    # 데이터 준비
    audio_files, labels = [], []
    emotions = ['기쁨', '슬픔', '분노', '불안', '상처', '당황', '중립']

    for i in range(4):
        audio_path = os.path.join(DATASET_DIR, 'TS1/1.감정', f"{i + 1}.{emotions[i]}")
        if not os.path.exists(audio_path):
            print(f"경로 없음: {audio_path}")
            continue
        
        dir_list = get_folder_names(audio_path)

        for j in dir_list:
            audio_dir = os.path.join(audio_path, j)
            file_list = get_file_names(audio_dir)
            
            for file in file_list:
                audio_files.append(os.path.join(audio_dir, file))
                labels.append(emotions[i])
    
    for i in range(3):
        audio_path = os.path.join(DATASET_DIR, 'TS2/1.감정', f"{i + 5}.{emotions[i + 4]}")
        if not os.path.exists(audio_path):
            print(f"경로 없음: {audio_path}")
            continue
        
        dir_list = get_folder_names(audio_path)

        for j in dir_list:
            audio_dir = os.path.join(audio_path, j)
            file_list = get_file_names(audio_dir)
            
            for file in file_list:
                audio_files.append(os.path.join(audio_dir, file))
                labels.append(emotions[i + 4])

    # 데이터셋 및 학습 구성
    dataset = EmotionDataset(audio_files, labels, feature_extractor, label_encoder_path=LABEL_ENCODER_PATH)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = EmotionTransformer(input_dim=768, num_classes=len(set(labels)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model_save_path = os.path.join(MODEL_PATH, DATE, "model.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    train_model(train_dataloader, model, criterion, optimizer, model_path=model_save_path, num_epochs=1, save_interval=1, batch_interval=10)

if __name__ == '__main__':
    main()
