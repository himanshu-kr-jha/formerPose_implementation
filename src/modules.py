import torch
import torch.nn as nn

# As described in section 3.1.3 and Fig 3
class MBFFN(nn.Module):
    """
    Multi-Branch Feed-Forward Network
    This replaces the standard FFN block in a Transformer.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        # Branch 1 with 3x3 Depth-wise Conv
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, 
                                 groups=hidden_features, bias=False)
        # Branch 2 with 5x5 Depth-wise Conv
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, 5, 1, 2, 
                                 groups=hidden_features, bias=False)
        
        self.mlp_in = nn.Linear(in_features, hidden_features * 2)
        self.gelu = nn.GELU()
        self.mlp_out = nn.Linear(hidden_features * 2, out_features)

    def forward(self, x, H, W):
        # x is (B, N, C) where N = H*W
        B, N, C = x.shape
        
        # Project to hidden dim and split for two branches
        x_proj = self.mlp_in(x) # (B, N, 2*hidden)
        x1, x2 = torch.chunk(x_proj, 2, dim=-1) # (B, N, hidden) each

        # Reshape for 2D convolution
        x1 = x1.permute(0, 2, 1).reshape(B, -1, H, W) # (B, hidden, H, W)
        x2 = x2.permute(0, 2, 1).reshape(B, -1, H, W) # (B, hidden, H, W)

        # Apply parallel depth-wise convs
        x1 = self.dwconv1(x1)
        x2 = self.dwconv2(x2)
        
        # Flatten back
        x1 = x1.reshape(B, -1, N).permute(0, 2, 1) # (B, N, hidden)
        x2 = x2.reshape(B, -1, N).permute(0, 2, 1) # (B, N, hidden)

        # Concat and project out
        x = torch.cat([x1, x2], dim=-1) # (B, N, 2*hidden)
        x = self.gelu(x)
        x = self.mlp_out(x) # (B, N, out_features)
        return x

class MRSA(nn.Module):
    """
    Multi-Resolution Self-Attention
    Uses spatial reduction on K and V, plus a local feature receptor.
    """
    def __init__(self, dim, num_heads, reduction_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        
        # Spatial reduction conv
        self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, 
                            stride=reduction_ratio)
        self.norm = nn.LayerNorm(dim)
        
        # DropKey (as mentioned in)
        # This is a simple interpretation. The paper [21] has a specific impl.
        self.drop_key = nn.Dropout(p=0.1) 

        # Additional local features receptor f(x) = DWConv5x5
        self.local_receptor = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Apply spatial reduction to create K and V
        x_kv = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_kv = self.sr(x_kv).reshape(B, C, -1).permute(0, 2, 1) # (B, N_reduced, C)
        x_kv = self.norm(x_kv)
        
        kv = self.kv(x_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # (B, num_heads, N_reduced, C_head)

        # Apply DropKey
        k = self.drop_key(k)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Add local features
        local_feat = self.local_receptor(x.permute(0, 2, 1).reshape(B, C, H, W))
        local_feat = local_feat.reshape(B, C, N).permute(0, 2, 1)
        
        x = x + local_feat
        return x

# As described in section 3.2 and Fig 4
class PFormerAttention(nn.Module):
    """
    Point Transformer block with explicit relative position bias
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        
        # MLP for encoding the relative position bias r_ij
        # r_ij = MLP( (p_i - p_j) / ||p_i - p_j|| )
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim // num_heads),
            nn.ReLU(),
            nn.Linear(dim // num_heads, dim // num_heads)
        )

    def forward(self, x, pos):
        # x is (B, N, C) -- point features
        # pos is (B, N, 3) -- point (x,y,z) coordinates
        B, N, C = x.shape
        
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, num_heads, N, C_head)

        # Calculate relative position
        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1) # (B, N, N, 3)
        
        # Normalize to get direction vector (add epsilon for stability)
        pos_norm = torch.norm(pos_diff, dim=-1, keepdim=True) + 1e-6
        pos_dir = pos_diff / pos_norm # (B, N, N, 3)
        
        # Calculate the relative position bias r_ij
        r_ij = self.pos_mlp(pos_dir) # (B, N, N, C_head)
        r_ij = r_ij.permute(0, 3, 1, 2).unsqueeze(1) # (B, 1, C_head, N, N)
        # This shape is tricky. Let's re-read Eq 7.
        # It's (q_i * k_j^T + r_ij) / sqrt(d_k)
        # This implies r_ij needs to be (B, num_heads, N, N)
        
        # Let's retry the bias MLP
        self.pos_mlp_head = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, self.num_heads)
        )
        
        # A more direct implementation of the bias r_ij
        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1) # (B, N, N, 3)
        pos_norm = torch.norm(pos_diff, dim=-1, keepdim=True) + 1e-6
        pos_dir = pos_diff / pos_norm
        
        r_ij_bias = self.pos_mlp_head(pos_dir) # (B, N, N, num_heads)
        r_ij_bias = r_ij_bias.permute(0, 3, 1, 2) # (B, num_heads, N, N)
        
        # Calculate attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add the relative position bias
        attn = attn + r_ij_bias 
        
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x

# As described in section 3.3 and Fig 5
class MSTF(nn.Module):
    """
    Multi-scale Transformer Fusion module
    """
    def __init__(self, in_dim_2d, in_dim_3d, out_dim):
        super().__init__()
        
        self.fusion_dim = in_dim_2d + in_dim_3d
        
        # Self-Attention Fusion
        self.sa_fusion = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=8, # You can tune this
            dim_feedforward=self.fusion_dim * 4,
            batch_first=True
        )
        
        # Dense Fusion
        # (F_key)_l and (F_global)_l
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Final projection
        # Input is [F_key, F_global] concatenated, so 2 * fusion_dim
        self.final_proj = nn.Linear(self.fusion_dim * 2, out_dim)
        
    def forward(self, f_color, f_point):
        # f_color: (B, N, C_color)
        # f_point: (B, N, C_point)
        # Note: N (number of points) must be the same. 
        # This requires processing (e.g., via FPS or projection)
        # to match the number of 2D patches and 3D points.
        
        # 1. Raw Fusion (Concatenation)
        f_raw = torch.cat([f_color, f_point], dim=-1) # (B, N, C_color + C_point)
        
        # 2. Self-Attention Fusion to get F_key
        f_key = self.sa_fusion(f_raw) # (B, N, fusion_dim)
        
        # 3. Dense Fusion
        # Get F_global
        f_global = self.global_pool(f_key.permute(0, 2, 1)) # (B, fusion_dim, 1)
        f_global = f_global.permute(0, 2, 1).repeat(1, f_key.shape[1], 1) # (B, N, fusion_dim)
        
        # Concatenate Key and Global features
        f_dense = torch.cat([f_key, f_global], dim=-1) # (B, N, 2 * fusion_dim)
        
        f_out = self.final_proj(f_dense)
        
        # The output is per-point/patch. This is then fed to the
        # final regression head (Fig 2d), which likely pools
        # these features before regressing the pose.
        return f_out
