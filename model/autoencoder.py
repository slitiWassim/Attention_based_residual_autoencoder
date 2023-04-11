import torch
from .reconstruction_model import Reconstruction3DEncoder, Reconstruction3DDecoder ,AttentionBasedResidual3DDecoder

class convAE(torch.nn.Module):
    def __init__(self):  # for reconstruction
        super(convAE, self).__init__()

        self.reconstruction = True

        self.encoder = Reconstruction3DEncoder(chnum_in=1)  # black and white
        self.decoder = Reconstruction3DDecoder(chnum_in=1)  # black and white

    def forward(self, x):
        fea = self.encoder(x)
        output = self.decoder(fea.clone())

        return output
        
class convAEAttention(torch.nn.Module):
    def __init__(self):  # for reconstruction
        super(convAEAttention, self).__init__()

        self.reconstruction = True

        self.encoder = Reconstruction3DEncoder(chnum_in=1)  # black and white
        self.decoder = AttentionBasedResidual3DDecoder(chnum_in=1)  # black and white

    def forward(self, x):
        fea = self.encoder(x)
        output = self.decoder(fea.clone())

        return output
