import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 입력: 1 x 512 x 512
            nn.Conv2d(1, 64, 4, stride=2, padding=1),    # 64 x 256 x 256
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128 x 128 x 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 256 x 64 x 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1), # 512 x 32 x 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1024, 4, stride=2, padding=1), # 1024 x 16 x 16
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # 입력: 1024 x 16 x 16
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1), # 512 x 32 x 32
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 256 x 64 x 64
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 128 x 128 x 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64 x 256 x 256
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),     # 1 x 512 x 512
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.decoder(x)

class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output