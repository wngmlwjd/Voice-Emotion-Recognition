import os
from datetime import datetime

DATE = datetime.today().strftime("%Y-%m-%d")

EMOTIONS = ['기쁨', '슬픔', '분노', '불안', '상처', '당황', '중립']
TRAIN_DATASET_DIR = "./dataset/015.감성 및 발화 스타일별 음성합성 데이터/01.데이터/1.Training/원천데이터"
TEST_DATASET_DIR = "./dataset/015.감성 및 발화 스타일별 음성합성 데이터/01.데이터/2.Validation/원천데이터/1.감정"
TRAIN_MODEL_PATH = os.path.join("./VER/trained_model/", DATE, "model.pth")
TEST_MODEL_PATH = "./VER/trained_model/latest/model.pth"
LATEST_MODEL_PATH = "./VER/trained_model/latest/model.pth"
LABEL_ENCODER_PATH = "./VER/trained_model/label_encoder.pkl"