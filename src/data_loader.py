# In src/data_loader.py

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import pickle     # For loading .pkl files
import open3d as o3d # For loading .ply models

def depth_to_point_cloud(depth_img, K):
    """
    Converts a depth image to a 3D point cloud.
    K is the 3x3 camera intrinsic matrix.
    """
    H, W = depth_img.shape
    v, u = np.indices((H, W))
    
    Z = depth_img
    X = (u - K[0, 2]) * Z / K[0, 0] 
    Y = (v - K[1, 2]) * Z / K[1, 1]
    
    point_cloud = np.stack((X, Y, Z), axis=-1)
    return point_cloud.reshape(-1, 3), np.stack((u, v), axis=-1).reshape(-1, 2)

class PoseDataset(Dataset):
    def __init__(self, data_root, object_id, num_points=1000, is_train=True):
        self.num_points = num_points
        self.data_dir = os.path.join(data_root, object_id)
        
        # --- 1. Load the list of training/testing image indices ---
        # The file list (train.txt or test.txt) is correct
        train_test_split = 'train.txt' if is_train else 'test.txt'
        split_file = os.path.join(self.data_dir, train_test_split)
        with open(split_file, 'r') as f:
            self.image_indices = [line.strip() for line in f.readlines()]
            
        # --- 2. Create full file paths ---
        # FIX: Use 'JPEGImages' folder, not 'rgb'
        # FIX: Assume .png for now, change to .jpg if it fails
        self.rgb_files = [os.path.join(self.data_dir, 'JPEGImages', f"{idx}.png") for idx in self.image_indices]
        
        # !!!!!!!!!!!! CRITICAL PROBLEM !!!!!!!!!!!!
        # !!  The 'ls' output does not show a 'depth' folder.
        # !!  I will create a path, but this will fail if the folder doesn't exist.
        # !!  PLEASE CHECK: What is your depth folder named?
        self.depth_files = [os.path.join(self.data_dir, 'depth', f"{idx}.png") for idx in self.image_indices]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        self.mask_files = [os.path.join(self.data_dir, 'mask', f"{idx}.png") for idx in self.image_indices]

        # --- 3. Load poses and camera intrinsics from .pkl file ---
        # FIX: Load from train.pkl/test.pkl, not gt.yml
        pkl_file = 'train.pkl' if is_train else 'test.pkl'
        pkl_path = os.path.join(self.data_dir, pkl_file)
        
        with open(pkl_path, 'rb') as f:
            all_data = pickle.load(f)
        
        # We need to extract poses and K for our image indices
        self.gt_poses = []
        self.all_K = {} # Store K matrix, probably only one
        
        for idx_str in self.image_indices:
            # Data in pkl might be stored by int or str index
            try:
                data = all_data[int(idx_str)]
            except KeyError:
                data = all_data[idx_str]
                
            R = data['cam_R_m2c'].astype(np.float32)
            T = data['cam_t_m2c'].astype(np.float32) / 1000.0 # Convert mm to m
            self.gt_poses.append((R, T))
            
            if 'cam_K' in data:
                self.K = data['cam_K'].astype(np.float32)
            
        if not hasattr(self, 'K'):
            print("ERROR: Camera matrix 'cam_K' not found in .pkl file.")
            # Fallback, this will probably be wrong
            self.K = np.array([[572.4, 0, 320], [0, 573.5, 240], [0, 0, 1]], dtype=np.float32)

        # --- 4. Load the 3D model ---
        # FIX: Load from object folder, not top-level 'models' folder
        model_file = os.path.join(self.data_dir, f"{object_id}.ply")
        pcd = o3d.io.read_point_cloud(model_file)
        # Scale model from mm to meters (assuming model is in mm)
        self.model_points = np.asarray(pcd.points).astype(np.float32) / 1000.0
        
        # --- 5. Check symmetry ---
        symmetric_objects = ['can', 'eggbox', 'glue']
        self.is_symmetric = object_id in symmetric_objects

        print(f"Loaded {len(self.image_indices)} samples for object {object_id}.")
        print(f"Object is symmetric: {self.is_symmetric}")

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        
        # --- 1. Load data for the given index ---
        rgb_img = cv2.imread(self.rgb_files[idx])
        if rgb_img is None:
            # Try .jpg if .png failed
            rgb_img = cv2.imread(self.rgb_files[idx].replace('.png', '.jpg'))
        
        depth_img = cv2.imread(self.depth_files[idx], cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print(f"FATAL ERROR: Could not read depth image: {self.depth_files[idx]}")
            # This will fail, but it's the only way forward
            depth_img = np.zeros((480, 640), dtype=np.float32)
        
        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_UNCHANGED)
        gt_R, gt_T = self.gt_poses[idx]
        
        # Convert depth from mm to meters (if it's in mm)
        if np.max(depth_img) > 100:
            depth_img = depth_img.astype(np.float32) / 1000.0
        
        # Apply mask to depth (keep only object points)
        if mask is not None:
            depth_img[mask == 0] = 0 

        # --- 2. Convert depth to point cloud ---
        full_pc, full_pixels = depth_to_point_cloud(depth_img, self.K) # (H*W, 3)
        
        # --- 3. Filter out zero-depth points ---
        non_zero_indices = (full_pc[:, 2] > 0)
        full_pc = full_pc[non_zero_indices]
        full_pixels = full_pixels[non_zero_indices]

        if len(full_pc) == 0:
            print(f"Warning: No valid points in sample {idx} (check depth path/format). Using fallback.")
            return self.__getitem__(np.random.randint(0, len(self) - 1)) # Return a random sample

        # --- 4. Sample N points ---
        if len(full_pc) < self.num_points:
            indices = np.random.choice(len(full_pc), self.num_points, replace=True)
        else:
            indices = np.random.choice(len(full_pc), self.num_points, replace=False)
        
        sampled_points = full_pc[indices]    # (N, 3)
        sampled_pixels = full_pixels[indices] # (N, 2)
        
        # --- 5. Convert to Tensors ---
        return {
            'rgb': torch.from_numpy(rgb_img).permute(2, 0, 1).float(), # (3, H, W)
            'points': torch.from_numpy(sampled_points).float(),       # (N, 3)
            'pixels': torch.from_numpy(sampled_pixels).float(),       # (N, 2)
            'gt_rotation': torch.from_numpy(gt_R).float(),            # (3, 3)
            'gt_translation': torch.from_numpy(gt_T).float(),        # (3)
            'model_points': torch.from_numpy(self.model_points).float(), # (M, 3)
            'is_symmetric': self.is_symmetric
        }