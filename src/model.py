# In src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules import MBFFN, MRSA, PFormerAttention, MSTF # Import our blocks

# Note: Implementing the full CFormer/PFormer backbones is complex.
# We'll use a simplified version based on Fig 2(b)
# For a real implementation, you'd build these as 4-stage pyramids.

class FormerPose(nn.Module):
    def __init__(self, num_points=1000, num_classes=21): # num_classes for YCB
        super().__init__()
        self.num_points = num_points
        
        # --- 1. CFormer (Color) Branch ---
        # We'll use a simple CNN as a placeholder for the CFormer backbone
        # A real implementation would use the MRSA/MBFFN blocks
        self.cformer_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), # Downsamples
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU() # Downsamples
        )
        self.cformer_dim = 256

        # --- 2. PFormer (Geometric) Branch ---
        # A simple PointNet-like encoder
        self.pformer_conv1 = nn.Conv1d(3, 64, 1)
        self.pformer_conv2 = nn.Conv1d(64, 128, 1)
        self.pformer_conv3 = nn.Conv1d(128, 256, 1)
        # We use a PFormerAttention block
        self.pformer_attn = PFormerAttention(dim=256, num_heads=4)
        self.pformer_dim = 256

        # --- 3. MSTF (Fusion) Module ---
        self.fusion_dim = self.cformer_dim + self.pformer_dim
        self.mstf = MSTF(
            in_dim_2d=self.cformer_dim, 
            in_dim_3d=self.pformer_dim,
            out_dim=self.fusion_dim
        )
        
        # --- 4. Regression Head --- [cite: 147-148]
        # This takes the final pooled features
        self.reg_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            # Output: 3 for translation (x,y,z) + 4 for rotation (quaternion)
            nn.Linear(256, 7) 
        )

    def forward(self, rgb, points, pixels):
        # rgb: (B, 3, H, W)
        # points: (B, N, 3)
        # pixels: (B, N, 2) -- (u,v) coordinates
        B, N, _ = points.shape
        H, W = rgb.shape[2], rgb.shape[3]

        # --- 1. CFormer Forward ---
        f_color_map = self.cformer_backbone(rgb) # (B, C_rgb, H', W')
        
        # --- 2. PFormer Forward ---
        f_point = points.permute(0, 2, 1) # (B, 3, N)
        f_point = F.relu(self.pformer_conv1(f_point))
        f_point = F.relu(self.pformer_conv2(f_point))
        f_point = F.relu(self.pformer_conv3(f_point)) # (B, C_pt, N)
        f_point = f_point.permute(0, 2, 1) # (B, N, C_pt)
        # Apply PFormer attention
        f_point = self.pformer_attn(f_point, points) # (B, N, C_pt)

        # --- 3. Dense Fusion ---
        # We need to sample 2D features at the 3D points' pixel locations
        
        # Normalize pixel coordinates from [0, W] to [-1, 1]
        norm_pixels = pixels.clone()
        norm_pixels[..., 0] = (pixels[..., 0] / (W - 1)) * 2 - 1
        norm_pixels[..., 1] = (pixels[..., 1] / (H - 1)) * 2 - 1
        norm_pixels = norm_pixels.unsqueeze(1) # (B, 1, N, 2)

        # F.grid_sample takes (B, C, H, W) and (B, N_out, 1, 2)
        # Our norm_pixels is (B, 1, N, 2), let's swap dims
        norm_pixels = norm_pixels.permute(0, 2, 1, 3) # (B, N, 1, 2)
        
        f_color_sampled = F.grid_sample(
            f_color_map, 
            norm_pixels, 
            mode='bilinear', 
            align_corners=True
        ) # (B, C_rgb, N, 1)
        
        f_color = f_color_sampled.squeeze(-1).permute(0, 2, 1) # (B, N, C_rgb)

        # --- 4. MSTF Forward ---
        # f_color is (B, N, 256)
        # f_point is (B, N, 256)
        f_fused = self.mstf(f_color, f_point) # (B, N, fusion_dim)
        
        # --- 5. Pooling & Regression ---
        # Average pool all N point features to get one global feature
        f_global = torch.mean(f_fused, dim=1) # (B, fusion_dim)
        
        pose_output = self.reg_head(f_global) # (B, 7)
        
        pred_T = pose_output[:, :3] # Translation
        pred_R_quat = pose_output[:, 3:] # Rotation (quaternion)
        
        # Normalize quaternion to be a unit quaternion
        pred_R_quat = F.normalize(pred_R_quat, p=2, dim=1)
        
        return pred_T, pred_R_quat