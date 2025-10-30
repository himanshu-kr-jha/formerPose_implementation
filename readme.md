FormerPose_Project/
├── datasets/
│   ├── linemod/
│   ├── ycb_video/
|
├── src/
│   ├── data_loader.py    # (Step 2) Our custom Dataset class
│   ├── modules.py        # (Step 3) MRSA, MBFFN, PFormerAttention, MSTF
│   ├── model.py          # (Step 3) The main FormerPose network
│   ├── loss.py           # (Step 4) The ADD(-S) loss function
│
├── train.py                # (Step 5) Our main training script
└── eval.py                 # (Step 6) Our evaluation script