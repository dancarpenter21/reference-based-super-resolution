import torch
import torch.nn as nn
import torch.nn.functional as F

class RefUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super(RefUNet, self).__init__()

        # Encoder for LR Image
        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        
        # Encoder for Reference Image
        self.ref_enc1 = self._block(in_channels, base_channels)
        self.ref_enc2 = self._block(base_channels, base_channels * 2)

        # Fusion Layer
        self.fusion = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)

        # Decoder / Upsampler
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = self._block(base_channels * 2, base_channels) # + skip connection from enc1
        
        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1) # Upscale 2x
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
        # Additional Upscaling (assuming 4x total upscaling needed)
        # The UNet structure above does 2x upsampling if input is smaller.
        # But here, we want: Input LR -> Output HR (4x bigger)
        # Standard UNet keeps size same or downscales/upscales back to original.
        # We need an "Upscaling" network.
        
        # Let's redesign:
        # 1. Feature Exitrain LR
        # 2. Feature Extrain Ref
        # 3. Concatenate
        # 4. Residual Blocks
        # 5. Upscale
        
        self.upscale_factor = 4
        
        # Re-defining layers for ResNet approach
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.ref_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_conv = nn.Conv2d(base_channels * 2, base_channels, 1)
        
        self.res_blocks = nn.Sequential(
            *[ResBlock(base_channels) for _ in range(8)]
        )
        
        self.upsampler = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(base_channels, out_channels, 3, 1, 1)
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, lr, ref):
        # Extract features
        lr_feat = self.feature_extractor(lr)
        ref_feat = self.ref_feature_extractor(ref)
        
        # Simple Fusion (Concat)
        # Note: Ref should ideally be aligned. Here we assume naive fusion for baseline.
        # Ensure Ref features are same size as LR features?
        # If Ref is HR, we need to downscale it to match LR feature size?
        # Or we act on LR size.
        
        # Assuming Ref is passed as "already aligned to LR" or global context.
        # If Ref is input as HR image (4x larger), we must downscale it or process it.
        # Let's assume Ref is downscaled to LR size for feature extraction simplicity 
        # OR we use a stride in ref_feature_extractor.
        
        if ref.shape[2:] != lr.shape[2:]:
            ref = F.interpolate(ref, size=lr.shape[2:], mode='bilinear', align_corners=False)
            
        ref_feat = self.ref_feature_extractor(ref)
        
        fused = torch.cat([lr_feat, ref_feat], dim=1)
        fused = self.fusion_conv(fused)
        
        res = self.res_blocks(fused)
        res = res + lr_feat # Global Residual
        
        out = self.upsampler(res)
        return out

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out
