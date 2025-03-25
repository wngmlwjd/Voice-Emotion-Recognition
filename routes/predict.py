from flask import Blueprint, request, jsonify
import os
import tempfile
import sys
import torch

from VER.module.test import test_model
from VER.module.recommendation import get_recommendations
from VER.module.load_model import load_model
from VER.module.feature_extraction import HuBERTFeatureExtractor

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    audio_files = request.files['audio']

    if audio_files:
        try:
            # temp 파일로 저장
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, audio_files.filename)
            audio_files.save(temp_path)

            # 모델 및 특성 추출기 로드
            model_path = "./VER/trained_model/latest/model.pth"
            label_encoder_path = "./VER/trained_model/label_encoder.pkl"
            feature_extractor = HuBERTFeatureExtractor()

            # 모델 로딩
            model = load_model(model_path, device="mps")

            # 예측 실행 (test_audio_files는 경로 리스트로 전달)
            predicted_labels, accuracy = test_model(
                test_audio_files=[temp_path],
                feature_extractor=feature_extractor,
                model=model,
                label_encoder_path=label_encoder_path,
                device="mps"
            )
            
            # finally:
            #     if os.path.exists(temp_path):
            #         os.remove(temp_path)

            # 결과 반환
            return jsonify({
                "predictions": predicted_labels,
                "accuracy": accuracy
            })

        except Exception as e:
            print(f"Error: {str(e)}") 
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': '파일 없음'}), 400