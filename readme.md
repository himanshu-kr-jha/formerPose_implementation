# FormerPose: Project Structure

This document outlines the file and folder structure for the FormerPose implementation project.

```
FormerPose_Project/
├── datasets/
│   ├── linemod/
│   ├── ycb_video/
│
├── src/
│   ├── data_loader.py    # (Step 2) Our custom Dataset class
│   ├── modules.py        # (Step 3) MRSA, MBFFN, PFormerAttention, MSTF
│   ├── model.py          # (Step 3) The main FormerPose network
│   ├── loss.py           # (Step 4) The ADD(-S) loss function
│
├── train.py                # (Step 5) Our main training script
└── eval.py                 # (Step 6) Our evaluation script
```

## 📁 File Descriptions

### `/datasets`

This directory holds the raw training and testing data.

  * `/linemod`: Contains the LineMOD dataset, 
  * `/ycb_video`: Contains the YCB-Video dataset.

### `/src`

This directory holds all the core Python source code for the project.

  * `data_loader.py`: Contains the PyTorch `Dataset` class. This file is responsible for loading images (RGB), depth maps, converting depth to point clouds, sampling points, and loading ground-truth poses.
  * `modules.py`: Contains the custom neural network building blocks described in the paper, such as `MRSA`, `MBFFN`, `PFormerAttention`, and `MSTF`.
  * `model.py`: Defines the main `FormerPose` network architecture, assembling the blocks from `modules.py`.
  * `loss.py`: Implements the specialized loss functions required for 6D pose estimation, primarily the **ADD(-S) loss**.

### Root Files

These are the main executable scripts.

  * `train.py`: The main script to start the training process. It initializes the dataset, model, and optimizer, and runs the training loop.
  * `eval.py`: A script to evaluate a trained model's performance on the test set, calculating metrics like `ADD(S)-0.1d`.