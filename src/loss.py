# In src/loss.py

import torch
import torch.nn as nn

def quaternion_to_matrix(quaternions):
    """
    Converts a batch of quaternions to rotation matrices.
    quaternions: (B, 4) -- (w, x, y, z) or (a, b, c, d)
    """
    a, b, c, d = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    B = quaternions.size(0)
    
    R = torch.zeros((B, 3, 3), device=quaternions.device)
    
    R[:, 0, 0] = a**2 + b**2 - c**2 - d**2
    R[:, 0, 1] = 2*b*c - 2*a*d
    R[:, 0, 2] = 2*b*d + 2*a*c
    
    R[:, 1, 0] = 2*b*c + 2*a*d
    R[:, 1, 1] = a**2 - b**2 + c**2 - d**2
    R[:, 1, 2] = 2*c*d - 2*a*b
    
    R[:, 2, 0] = 2*b*d - 2*a*c
    R[:, 2, 1] = 2*c*d + 2*a*b
    R[:, 2, 2] = a**2 - b**2 - c**2 + d**2
    
    return R

def compute_add_s_loss(pred_R, pred_T, gt_R, gt_T, model_points, is_symmetric):
    """
    Calculates the ADD(-S) loss.
    pred_R, gt_R: (B, 3, 3) rotation matrices
    pred_T, gt_T: (B, 3) translation vectors
    model_points: (B, M, 3) 3D model points
    is_symmetric: (B) boolean tensor
    """
    B = pred_R.shape[0]
    M = model_points.shape[1]
    
    # Transform points
    pred_points = torch.bmm(model_points, pred_R.transpose(1, 2)) + pred_T.unsqueeze(1)
    gt_points = torch.bmm(model_points, gt_R.transpose(1, 2)) + gt_T.unsqueeze(1)
    
    # Calculate error
    losses = torch.zeros(B, device=pred_R.device)
    
    for i in range(B):
        if not is_symmetric[i]:
            # ADD Loss [cite: 331]
            error = torch.norm(pred_points[i] - gt_points[i], dim=-1) # (M)
            losses[i] = torch.mean(error)
        else:
            # ADD-S Loss [cite: 333]
            # (1, M, M) matrix of pairwise distances
            dists = torch.cdist(pred_points[i].unsqueeze(0), gt_points[i].unsqueeze(0))
            min_dists, _ = torch.min(dists, dim=2) # (1, M)
            losses[i] = torch.mean(min_dists)
            
    return torch.mean(losses)