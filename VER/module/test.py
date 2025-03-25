from torch.utils.data import DataLoader
from .evaluation import evaluate_model, calculate_accuracy
from .dataset import EmotionDataset, collate_fn
from .feature_extraction import HuBERTFeatureExtractor
from .utils import load_label_encoder
import torch

def test_model(test_dataloader, model, dataset):
    model.eval()
    predictions = []
    ground_truth = []
    device = torch.device("mps")

    with torch.no_grad():
        for features, labels in test_dataloader:
            features = features.to(device)
            labels = labels.to(device)

            # 모델에 입력
            features = features.permute(1, 0, 2)

            outputs = model(features)

            predicted_labels = torch.argmax(outputs, dim=1)
            predictions.extend(predicted_labels.cpu().tolist())
            ground_truth.extend(labels.cpu().tolist())

    # 예측 결과 반환
    return predictions, ground_truth