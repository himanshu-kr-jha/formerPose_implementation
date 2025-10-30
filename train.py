# In train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_loader import PoseDataset
from src.model import FormerPose
from src.loss import compute_add_s_loss, quaternion_to_matrix

def main():
    # --- 1. Config ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # !!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!
    # !!  Update this to your LineMOD folder path
    DATA_ROOT = "/home/himanshu-kumar-jha/Documents/FormerPose/datasets/LINEMOD" 
    #
    # !!  Choose which object to train on (e.g., '000001' for ape)
    OBJECT_ID = "phone" 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    NUM_POINTS = 1000
    BATCH_SIZE = 4
    EPOCHS = 50 
    LEARNING_RATE = 1e-4

    # --- 2. Data ---
    print(f"Loading training data for object: {OBJECT_ID}")
    train_dataset = PoseDataset(DATA_ROOT, OBJECT_ID, num_points=NUM_POINTS, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- 3. Model ---
    model = FormerPose(num_points=NUM_POINTS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    # --- 4. Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(train_loader):
            # Move data to device
            rgb = batch['rgb'].to(DEVICE)
            points = batch['points'].to(DEVICE)
            pixels = batch['pixels'].to(DEVICE)
            gt_R = batch['gt_rotation'].to(DEVICE)
            gt_T = batch['gt_translation'].to(DEVICE)
            model_pts = batch['model_points'].to(DEVICE)
            is_sym = batch['is_symmetric'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_T, pred_R_quat = model(rgb, points, pixels)
            
            # Convert predicted quaternion to rotation matrix
            pred_R = quaternion_to_matrix(pred_R_quat)
            
            # Calculate loss
            loss = compute_add_s_loss(pred_R, pred_T, gt_R, gt_T, model_pts, is_sym)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Complete --- Average Loss: {avg_loss:.4f} ---")

    print("Training complete.")
    torch.save(model.state_dict(), f"formerpose_{OBJECT_ID}.pth")
    print(f"Model saved to formerpose_{OBJECT_ID}.pth")

if __name__ == "__main__":
    main()