from transformers import Wav2Vec2FeatureExtractor, HubertModel
import torchaudio
import torch

class HuBERTFeatureExtractor:
    def __init__(self, model_name="facebook/hubert-base-ls960"):
        # HuBERT 모델과 프로세서 초기화
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.model.eval()  # 평가 모드로 설정

    def load_audio(self, audio_file):
        # 오디오 파일 로드
        waveform, sample_rate = torchaudio.load(audio_file)
        
        return waveform, sample_rate

    def preprocess_audio(self, waveform, sample_rate, target_sample_rate=16000, max_length=10):
        # 모노로 변환
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 샘플링 레이트 변환
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            
        max_samples = target_sample_rate * max_length
        if waveform.size(1) > max_samples:
            waveform = waveform[:, :max_samples]
            
        return waveform

    def extract_features(self, audio_file):
        # 오디오 로드 및 전처리
        waveform, sample_rate = self.load_audio(audio_file)
        waveform = self.preprocess_audio(waveform, sample_rate)
        
        # 입력 형태 확인 및 조정
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2:
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(0)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        
        # 특성 추출
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values
        
        # 불필요한 차원 제거
        input_values = input_values.squeeze(1)  # (batch_size=1, sequence_length)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            
        features = outputs.last_hidden_state
        
        return features