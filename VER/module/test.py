from torch.utils.data import DataLoader
from .evaluation import evaluate_model, calculate_accuracy
from .dataset import EmotionDataset, collate_fn
from .feature_extraction import HuBERTFeatureExtractor
from .utils import load_label_encoder

def test_model(test_audio_files, feature_extractor, model, label_encoder_path, batch_size=32, device="mps"):
    # 데이터 준비
    label_encoder = load_label_encoder(label_encoder_path)
    dataset = EmotionDataset(test_audio_files, feature_extractor, label_encoder)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for features, labels in test_dataloader:
            features = features.permute(1, 0, 2)
            outputs = evaluate_model(model, features, device)

            predicted_labels = torch.argmax(outputs, dim=1)
            predictions.extend(predicted_labels.cpu().tolist())
            ground_truth.extend(labels.cpu().tolist())

    accuracy = calculate_accuracy(predictions, ground_truth)
    predicted_labels = label_encoder.inverse_transform(predictions)
    return predicted_labels, accuracy

