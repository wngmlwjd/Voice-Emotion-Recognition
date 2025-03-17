import torch.nn as nn

class EmotionTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmotionTransformer, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=16),
            num_layers=8
        )
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x, src_key_padding_mask=None):
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = x.mean(dim=0)
        return self.fc(x)
