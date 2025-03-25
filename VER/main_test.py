from module.feature_extraction import HuBERTFeatureExtractor
from module.dataset import EmotionDataset, collate_fn
from module.config import TEST_DATASET_DIR, DATE, LABEL_ENCODER_PATH, TEST_MODEL_PATH
from module.test import test_model
from module.model import EmotionTransformer

import torch
from torch.utils.data import DataLoader
import os


def main():
    # 테스트 데이터 준비
    test_audio_files = [
        os.path.join(TEST_DATASET_DIR, "1.기쁨/0029_G2A4E1S0C0_KJE/0029_G2A4E1S0C0_KJE_000001.wav"),
        # os.path.join(TEST_DATASET_DIR, "2.슬픔/0033_G2A3E2S0C0_KMA/0033_G2A3E2S0C0_KMA_000020.wav"),
        os.path.join(TEST_DATASET_DIR, "3.분노/0018_G2A3E3S0C0_JBR/0018_G2A3E3S0C0_JBR_000019.wav"),
        # os.path.join(TEST_DATASET_DIR, "4.불안/0012_G1A2E4S0C0_CHY/0012_G1A2E4S0C0_CHY_000011.wav"),
        os.path.join(TEST_DATASET_DIR, "5.상처/0005_G1A3E5S0C0_LJB/0005_G1A3E5S0C0_LJB_000014.wav"),
        os.path.join(TEST_DATASET_DIR, "6.당황/0020_G2A4E6S0C0_HGW/0020_G2A4E6S0C0_HGW_000009.wav"),
        os.path.join(TEST_DATASET_DIR, "7.중립/0044_G2A5E7S0C0_KTH/0044_G2A5E7S0C0_KTH_000012.wav")
    ]
    # test_labels = ["기쁨", "슬픔", "분노", "불안", "상처", "당황", "중립"]  # 정답 레이블
    test_labels = ["기쁨", "분노", "상처", "당황", "중립"]  # 정답 레이블
    
    feature_extractor = HuBERTFeatureExtractor()
    dataset = EmotionDataset(test_audio_files, test_labels, feature_extractor, LABEL_ENCODER_PATH)
    test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("mps")
    model = torch.load(TEST_MODEL_PATH, weights_only=False)
    model = model.to(device)
    
    predictions, ground_truth = test_model(test_dataloader, model, dataset)
    
    predicted_labels = dataset.label_encoder.inverse_transform(predictions)
    ground_truth_labels = dataset.label_encoder.inverse_transform(ground_truth)
    
    correct_predictions = sum([1 for p, g in zip(predictions, ground_truth) if p == g])
    accuracy = correct_predictions / len(predictions)
    
    print(f"Predictions: {predicted_labels}")
    print(f"Ground Truth: {ground_truth_labels}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
