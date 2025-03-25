import torch
from VER.module.model import EmotionTransformer  

def load_model(model_path, device="mps"):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model    