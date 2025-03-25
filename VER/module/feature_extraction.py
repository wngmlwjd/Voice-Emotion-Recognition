from transformers import Wav2Vec2FeatureExtractor, HubertModel
import torchaudio
import torch
import soundfile

class HuBERTFeatureExtractor:
    def __init__(self, model_name="facebook/hubert-base-ls960"):
    # def __init__(self, model_name="team-lucid/hubert-xlarge-korean"):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, force_download=True)
        self.model = HubertModel.from_pretrained(model_name, force_download=True)
        self.model.eval()

    def load_audio(self, audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        return waveform, sample_rate

    def preprocess_audio(self, waveform, sample_rate, target_sample_rate=16000, max_length=10):
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        max_samples = target_sample_rate * max_length
        if waveform.size(1) > max_samples:
            waveform = waveform[:, :max_samples]
        return waveform

    def extract_features(self, audio_file):
        waveform, sample_rate = self.load_audio(audio_file)
        waveform = self.preprocess_audio(waveform, sample_rate)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.squeeze(1)
        with torch.no_grad():
            outputs = self.model(input_values)
        return outputs.last_hidden_state
