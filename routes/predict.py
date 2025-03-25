from flask import Blueprint, request, jsonify
import os
import tempfile
import sys
import torch
from torch.utils.data import DataLoader

from VER.module.feature_extraction import HuBERTFeatureExtractor
from VER.module.dataset import EmotionDataset, collate_fn
from VER.module.config import TEST_DATASET_DIR, DATE, LABEL_ENCODER_PATH, TEST_MODEL_PATH
from VER.module.test import test_model
from VER.module.model import EmotionTransformer

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    audio_files = request.files['audio']

    if audio_files:
        try:
            feature_extractor = HuBERTFeatureExtractor()
            dataset = EmotionDataset([audio_files], ["중립"], feature_extractor, LABEL_ENCODER_PATH)
            test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
            
            device = torch.device("mps")
            model = torch.load(TEST_MODEL_PATH, weights_only=False)
            model = model.to(device)
    
            predictions, _ = test_model(test_dataloader, model, dataset)
    
            predicted_labels = dataset.label_encoder.inverse_transform(predictions).tolist()

            # 결과 반환
            return jsonify({
                "predictions": predicted_labels,
            })

        except Exception as e:
            print(f"Error: {str(e)}") 
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': '파일 없음'}), 400