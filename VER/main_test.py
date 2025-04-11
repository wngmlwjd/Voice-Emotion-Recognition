from module.feature_extraction import HuBERTFeatureExtractor
from module.dataset import EmotionDataset, collate_fn
from module.config import TEST_DATASET_DIR, DATE, LABEL_ENCODER_PATH, TEST_MODEL_PATH, EMOTIONS
from module.test import test_model
from module.model import EmotionTransformer
from module.utils import get_file_names, get_folder_names

import torch
from torch.utils.data import DataLoader
import os
from collections import defaultdict

def main():
    test_audio_files, test_labels = [], []
    
    for i in range(7):
        audio_path = os.path.join(TEST_DATASET_DIR, f"{i + 1}.{EMOTIONS[i]}")
        if not os.path.exists(audio_path):
            print(f"경로 없음: {audio_path}")
            continue
        
        dir_list = get_folder_names(audio_path)

        for j in dir_list:
            audio_dir = os.path.join(audio_path, j)
            file_list = get_file_names(audio_dir)
            
            for file in file_list:
                test_audio_files.append(os.path.join(audio_dir, file))
                test_labels.append(EMOTIONS[i])
    
    feature_extractor = HuBERTFeatureExtractor()
    dataset = EmotionDataset(test_audio_files, test_labels, feature_extractor, LABEL_ENCODER_PATH)
    test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("mps")
    model = torch.load(TEST_MODEL_PATH, weights_only=False)
    model = model.to(device)
    
    predictions, ground_truth = test_model(test_dataloader, model, dataset)
    
    predicted_labels = dataset.label_encoder.inverse_transform(predictions)
    ground_truth_labels = dataset.label_encoder.inverse_transform(ground_truth)
    
    # 전체 정확도 계산
    correct_predictions = sum([1 for p, g in zip(predicted_labels, ground_truth_labels) if p == g])
    accuracy = correct_predictions / len(predicted_labels)
    
    # 감정별 정확도 계산
    emotion_correct = defaultdict(int)
    emotion_total = defaultdict(int)
    
    for p, g in zip(predicted_labels, ground_truth_labels):
        emotion_total[g] += 1
        if p == g:
            emotion_correct[g] += 1
    
    print("각 감정별 정확도:")
    for emotion in EMOTIONS:
        if emotion_total[emotion] > 0:
            emotion_acc = emotion_correct[emotion] / emotion_total[emotion]
            print(f"{emotion}: {emotion_acc:.4f} ({emotion_correct[emotion]}/{emotion_total[emotion]})")
    
    print(f"전체 정확도: {accuracy:.4f}")

if __name__ == '__main__':
    main()
