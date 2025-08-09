from torch import nn
from .encoder import ViTEncoder
from .decoder import PaletteDecoder
from .attention import create_padding_mask, create_causal_mask

class ArC(nn.Module):
    def __init__(self, **kwargs):
        super(ArC, self).__init__()
        self.encoder = ViTEncoder(**kwargs)
        self.decoder = PaletteDecoder(**kwargs)

    def forward(self, image, palette):
        B,C,H,W = image.shape
        num_colors = palette.shape[1]
        padding_mask = create_padding_mask(palette).to(device=image.device)
        casual_mask = create_causal_mask(B, num_colors).to(device=image.device)
        combined_mask = padding_mask | casual_mask
        encoding = self.encoder(image, palette, mask=padding_mask)
        logits = self.decoder( palette,encoding, combined_mask)
        return logits