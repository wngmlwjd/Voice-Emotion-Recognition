import torch

def evaluate_model(model, features, device):
    """
    모델에 데이터를 입력하고 예측 결과를 반환하는 함수.
    """
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        outputs = model(features)
        return outputs

"""
예측값과 실제값을 비교하여 정확도를 계산하는 함수.
"""
def calculate_accuracy(predictions, ground_truth):
    correct = sum([1 if pred == true else 0 for pred, true in zip(predictions, ground_truth)])
    accuracy = correct / len(ground_truth)
    return accuracy
